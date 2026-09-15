"""Spectrum estimation: complex IQ samples -> relative power spectrum in dB.

This module is pure signal processing. It never touches the SDR and never
plots; callers decide what to do with the arrays it returns.

The returned power is *relative* dB: the ADALM-PLUTO is not amplitude
calibrated, so only differences between bins are meaningful, not absolute
levels.

Multi-capture estimates (average, max hold) are combined in the *linear*
power domain and converted to dB only at the very end.

Unreliable bins (center-frequency artifact, filter roll-off at the edges) are
described by a boolean validity mask; spectra themselves are never altered.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional

import numpy as np

from . import config


class Spectrum(NamedTuple):
    """One power-spectrum estimate.

    Attributes:
        frequencies_hz: Absolute RF frequency of each bin, ascending.
        power_db: Relative power of each bin, in dB.
    """

    frequencies_hz: np.ndarray
    power_db: np.ndarray


class AverageMaxHoldSpectrum(NamedTuple):
    """Average and max-hold estimates computed from the same captures.

    Attributes:
        frequencies_hz: Absolute RF frequency of each bin, ascending.
        average_power_db: Mean linear power per bin, in relative dB.
        max_hold_power_db: Maximum linear power per bin, in relative dB.
    """

    frequencies_hz: np.ndarray
    average_power_db: np.ndarray
    max_hold_power_db: np.ndarray


def validate_iq(samples: np.ndarray) -> np.ndarray:
    """Check that ``samples`` is a usable 1-D complex IQ buffer.

    Returns:
        The samples as a complex128 array.

    Raises:
        ValueError: if the buffer is empty, not 1-D, or contains NaN/Inf.
    """
    array = np.asarray(samples)

    if array.size == 0:
        raise ValueError("Empty IQ buffer: no samples were received.")
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D IQ buffer, got shape {array.shape}.")

    array = array.astype(np.complex128, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError("IQ buffer contains NaN or Inf values.")

    return array


def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    """Subtract the mean of the buffer.

    The Pluto's direct-conversion receiver leaks LO energy into the baseband,
    which appears as a constant term and therefore as a spike at the centre
    frequency. Subtracting the mean removes it.
    """
    return samples - np.mean(samples)


def frequency_axis(
    center_frequency: float,
    sample_rate: float,
    fft_size: int,
) -> np.ndarray:
    """Return the absolute RF frequency (Hz) of each FFT bin, ascending.

    The axis is ``fftshift``-ordered so it lines up with a shifted spectrum.
    """
    baseband = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    return center_frequency + baseband


def get_valid_spectrum_mask(
    frequencies_hz: np.ndarray,
    center_frequency_hz: float,
    sample_rate_hz: float,
    dc_exclusion_hz: float = config.DC_EXCLUSION_HZ,
    edge_exclusion_hz: float = config.EDGE_EXCLUSION_HZ,
) -> np.ndarray:
    """Boolean mask of the bins that are reliable for analysis and stitching.

    A bin is invalid (``False``) if it lies in any of:
      A. ``center_frequency_hz +/- dc_exclusion_hz`` (LO leakage / DC residue)
      B. the lowest ``edge_exclusion_hz`` of the capture (filter roll-off)
      C. the highest ``edge_exclusion_hz`` of the capture (filter roll-off)

    The capture spans ``center_frequency_hz +/- sample_rate_hz / 2``. The
    spectrum itself is not touched: apply the mask where it is needed, e.g.
    ``power_db[mask]``.

    Args:
        frequencies_hz: Absolute RF frequency of each bin (from
            :func:`frequency_axis`).
        center_frequency_hz: LO frequency the capture was tuned to.
        sample_rate_hz: Sample rate of the capture, i.e. its observed span.
        dc_exclusion_hz: Half-width of the excluded region around the center.
        edge_exclusion_hz: Width excluded at each edge of the capture.

    Returns:
        Boolean array with the same shape as ``frequencies_hz``.

    Raises:
        ValueError: if the exclusions are negative or leave no valid bins.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array.")
    if dc_exclusion_hz < 0 or edge_exclusion_hz < 0:
        raise ValueError("Exclusion widths must not be negative.")

    offset = frequencies - center_frequency_hz
    half_span = sample_rate_hz / 2.0

    # A width of 0 disables that exclusion entirely (keeps the exact-center bin).
    outside_dc = (np.abs(offset) > dc_exclusion_hz) | (dc_exclusion_hz == 0)
    above_lower_edge = offset >= -half_span + edge_exclusion_hz
    below_upper_edge = offset <= half_span - edge_exclusion_hz

    mask = outside_dc & above_lower_edge & below_upper_edge
    if not np.any(mask):
        raise ValueError(
            f"No valid bins left: DC exclusion +/-{dc_exclusion_hz} Hz and edge "
            f"exclusion {edge_exclusion_hz} Hz cover the whole "
            f"{sample_rate_hz} Hz capture."
        )
    return mask


def power_to_db(
    power_linear: np.ndarray,
    epsilon: float = config.POWER_EPSILON,
) -> np.ndarray:
    """Convert linear power to relative dB. ``epsilon`` prevents log10(0)."""
    return 10.0 * np.log10(power_linear + epsilon)


def compute_linear_power(
    samples: np.ndarray,
    fft_size: Optional[int] = None,
) -> np.ndarray:
    """Linear power per FFT bin for one IQ buffer (``fftshift``-ordered).

    Pipeline: validate -> remove DC -> Hann window -> FFT -> fftshift ->
    ``|X|**2``.

    The result is divided by ``fft_size * sum(window**2)``. That is the same
    fixed constant for every capture of the same size, so it keeps levels
    independent of the window and FFT length without normalising captures
    against each other.

    Args:
        samples: Complex baseband samples from the SDR.
        fft_size: FFT length. Defaults to the number of samples. A larger
            value zero-pads, a smaller one truncates.

    Raises:
        ValueError: if the IQ buffer or the FFT size is invalid.
    """
    iq = validate_iq(samples)

    if fft_size is None:
        fft_size = iq.size
    if fft_size <= 0:
        raise ValueError(f"FFT size must be positive, got {fft_size}.")

    iq = remove_dc_offset(iq)

    if fft_size < iq.size:
        iq = iq[:fft_size]

    # Hann window suppresses spectral leakage from the abrupt buffer edges.
    window = np.hanning(iq.size)
    spectrum = np.fft.fftshift(np.fft.fft(iq * window, n=fft_size))

    window_energy = np.sum(window ** 2)
    return (np.abs(spectrum) ** 2) / (fft_size * window_energy)


def compute_power_spectrum(
    samples: np.ndarray,
    center_frequency: float = config.CENTER_FREQUENCY,
    sample_rate: float = config.SAMPLE_RATE,
    fft_size: Optional[int] = None,
    epsilon: float = config.POWER_EPSILON,
) -> Spectrum:
    """Estimate the power spectrum of a single complex IQ buffer.

    Args:
        samples: Complex baseband samples from the SDR.
        center_frequency: RF frequency the SDR was tuned to, in Hz.
        sample_rate: Sample rate used for the capture, in samples/second.
        fft_size: FFT length. Defaults to the number of samples.
        epsilon: Added before log10() so silent bins cannot produce -inf.

    Returns:
        A :class:`Spectrum` with the frequency axis in Hz and relative power
        in dB.

    Raises:
        ValueError: if the IQ buffer or the FFT size is invalid.
    """
    power = compute_linear_power(samples, fft_size)
    frequencies = frequency_axis(center_frequency, sample_rate, power.size)
    return Spectrum(frequencies_hz=frequencies, power_db=power_to_db(power, epsilon))


def compute_average_and_max_hold(
    captures: Iterable[np.ndarray],
    center_frequency: float = config.CENTER_FREQUENCY,
    sample_rate: float = config.SAMPLE_RATE,
    fft_size: Optional[int] = None,
    epsilon: float = config.POWER_EPSILON,
) -> AverageMaxHoldSpectrum:
    """Combine several IQ buffers into average and max-hold spectra.

    Each capture is converted to linear power with the same pipeline as
    :func:`compute_power_spectrum` (same Hann window, same FFT size). Per bin,
    the linear powers are then averaged and maximised across captures, and
    only the two final results are converted to dB.

    Args:
        captures: IQ buffers, all with the same number of samples.
        center_frequency: RF frequency the SDR was tuned to, in Hz.
        sample_rate: Sample rate used for the captures, in samples/second.
        fft_size: FFT length. Defaults to the sample count of the first
            capture.
        epsilon: Added before log10() so silent bins cannot produce -inf.

    Returns:
        An :class:`AverageMaxHoldSpectrum`.

    Raises:
        ValueError: if there are no captures, a capture is invalid, or the
            captures do not all have the same number of samples.
    """
    average_linear, max_hold_linear = compute_average_and_max_hold_linear(
        captures, fft_size
    )
    return AverageMaxHoldSpectrum(
        frequencies_hz=frequency_axis(center_frequency, sample_rate, average_linear.size),
        average_power_db=power_to_db(average_linear, epsilon),
        max_hold_power_db=power_to_db(max_hold_linear, epsilon),
    )


def compute_average_and_max_hold_linear(
    captures: Iterable[np.ndarray],
    fft_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean and maximum of the linear power of several IQ buffers.

    Same pipeline and validation as :func:`compute_average_and_max_hold`, but
    the results stay in the linear power domain (``fftshift``-ordered).

    Returns:
        ``(average_power_linear, max_hold_power_linear)``.

    Raises:
        ValueError: if there are no captures, a capture is invalid, or the
            captures do not all have the same number of samples.
    """
    power_sum: Optional[np.ndarray] = None
    power_max: Optional[np.ndarray] = None
    expected_samples: Optional[int] = None
    count = 0

    for index, samples in enumerate(captures, start=1):
        samples = np.asarray(samples)
        if expected_samples is None:
            expected_samples = samples.size
        elif samples.size != expected_samples:
            raise ValueError(
                f"Capture {index}: expected {expected_samples} samples like "
                f"capture 1, got {samples.size}. All captures must share one "
                "FFT size and frequency axis."
            )

        try:
            power = compute_linear_power(samples, fft_size)
        except ValueError as exc:
            raise ValueError(f"Capture {index}: {exc}") from exc

        if power_sum is None:
            power_sum = power.copy()
            power_max = power.copy()
        else:
            power_sum += power
            np.maximum(power_max, power, out=power_max)
        count += 1

    if power_sum is None or power_max is None:
        raise ValueError("No captures were provided.")

    return power_sum / count, power_max
