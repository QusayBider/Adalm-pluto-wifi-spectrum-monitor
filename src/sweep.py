"""Full-band sweep: step the Pluto LO across the band and stitch the segments.

This module contains no FFT code of its own (see spectrum.py) and no plotting.

Per LO center frequency:
    tune -> settle -> discard warm-up buffers -> capture N buffers ->
    average / max hold in linear power -> valid-bin mask -> SweepSegment

Stitching:
    The valid bins of every segment are binned onto one common grid of
    ``resolution_hz`` wide cells spanning [band_start_hz, band_stop_hz).
    Inside a cell a segment contributes the mean linear power of its valid
    bins. Across segments the contributions are averaged (average spectrum)
    or maximised (max hold). Cells that no segment covers stay NaN; they are
    never filled with made-up values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from . import config
from .pluto_receiver import PlutoConnectionError, PlutoReceiver
from .spectrum import (
    compute_average_and_max_hold_linear,
    frequency_axis,
    get_valid_spectrum_mask,
    power_to_db,
)

HZ_PER_MHZ = 1e6

# Largest accepted difference between configured and read-back RX gain.
GAIN_TOLERANCE_DB = 0.5


class SweepError(RuntimeError):
    """A sweep step failed. The message names the center frequency if known.

    Attributes:
        center_frequency_hz: LO center of the failing segment, or ``None`` if
            the problem is not specific to one segment.
    """

    def __init__(self, message: str, center_frequency_hz: Optional[int] = None) -> None:
        super().__init__(message)
        self.center_frequency_hz = center_frequency_hz


@dataclass(frozen=True)
class SweepSegment:
    """One LO position of the sweep, before stitching.

    All arrays are ``fftshift``-ordered and have the FFT size as length. The
    power arrays are complete; ``valid_mask`` says which bins may be used.
    """

    center_frequency_hz: int
    frequencies_hz: np.ndarray
    average_power_linear: np.ndarray
    max_hold_power_linear: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class StitchedSpectrum:
    """Full-band result on the common grid.

    Attributes:
        frequencies_hz: Center frequency of each grid cell.
        average_power_linear: Stitched average, NaN where uncovered.
        max_hold_power_linear: Stitched max hold, NaN where uncovered.
        segment_count: Number of segments that contributed to each cell.
        resolution_hz: Width of one grid cell.
    """

    frequencies_hz: np.ndarray
    average_power_linear: np.ndarray
    max_hold_power_linear: np.ndarray
    segment_count: np.ndarray
    resolution_hz: float

    @property
    def average_power_db(self) -> np.ndarray:
        return power_to_db(self.average_power_linear)

    @property
    def max_hold_power_db(self) -> np.ndarray:
        return power_to_db(self.max_hold_power_linear)

    @property
    def covered_mask(self) -> np.ndarray:
        return self.segment_count > 0

    @property
    def uncovered_bins(self) -> int:
        return int(np.count_nonzero(~self.covered_mask))

    @property
    def coverage_percent(self) -> float:
        return 100.0 * np.count_nonzero(self.covered_mask) / self.segment_count.size

    def uncovered_ranges_hz(self) -> list[tuple[float, float]]:
        """``(start_hz, stop_hz)`` of every contiguous run of uncovered cells."""
        uncovered = np.concatenate(([0], (~self.covered_mask).astype(np.int8), [0]))
        changes = np.diff(uncovered)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1) - 1
        half = self.resolution_hz / 2.0
        return [
            (self.frequencies_hz[a] - half, self.frequencies_hz[b] + half)
            for a, b in zip(starts, stops)
        ]


# -- stitching ---------------------------------------------------------------


def make_band_grid(
    band_start_hz: float,
    band_stop_hz: float,
    resolution_hz: float,
) -> np.ndarray:
    """Center frequencies of the grid cells covering [start, stop).

    Raises:
        ValueError: if the span is not a positive whole number of cells.
    """
    span = band_stop_hz - band_start_hz
    if span <= 0 or resolution_hz <= 0:
        raise ValueError("Band stop must be above band start and resolution positive.")
    num_cells = int(round(span / resolution_hz))
    if num_cells < 1 or not np.isclose(num_cells * resolution_hz, span):
        raise ValueError(
            f"Band span {span} Hz is not a whole number of {resolution_hz} Hz cells."
        )
    return band_start_hz + (np.arange(num_cells) + 0.5) * resolution_hz


def _check_segment(segment: SweepSegment) -> None:
    """Raise ValueError if a segment's arrays are inconsistent or unusable."""
    label = f"Segment at {segment.center_frequency_hz / HZ_PER_MHZ:.1f} MHz"
    size = np.asarray(segment.frequencies_hz).size
    arrays = {
        "frequencies_hz": segment.frequencies_hz,
        "average_power_linear": segment.average_power_linear,
        "max_hold_power_linear": segment.max_hold_power_linear,
        "valid_mask": segment.valid_mask,
    }
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.ndim != 1 or array.size != size or size == 0:
            raise ValueError(
                f"{label}: {name} has shape {array.shape}, expected ({size},) "
                "like frequencies_hz."
            )
    if np.asarray(segment.valid_mask).dtype != np.bool_:
        raise ValueError(f"{label}: valid_mask must be a boolean array.")
    if not np.any(segment.valid_mask):
        raise ValueError(f"{label}: no valid bins.")
    for name in ("average_power_linear", "max_hold_power_linear"):
        valid_power = np.asarray(arrays[name])[segment.valid_mask]
        if not np.all(np.isfinite(valid_power)) or np.any(valid_power < 0):
            raise ValueError(f"{label}: {name} has NaN, Inf or negative valid bins.")


def segment_to_grid(
    segment: SweepSegment,
    band_start_hz: float = config.BAND_START_HZ,
    band_stop_hz: float = config.BAND_STOP_HZ,
    resolution_hz: float = config.STITCH_RESOLUTION_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin the valid bins of one segment onto the band grid.

    Only bins where ``valid_mask`` is True and that lie inside the band are
    used. Each cell gets the mean (average spectrum) and the maximum (max
    hold) of the linear power of the bins that fall into it.

    Returns:
        ``(average_power_linear, max_hold_power_linear)`` per grid cell, NaN
        where this segment has no valid bins.

    Raises:
        ValueError: if the segment is inconsistent (see :func:`_check_segment`).
    """
    _check_segment(segment)
    num_cells = make_band_grid(band_start_hz, band_stop_hz, resolution_hz).size

    mask = segment.valid_mask
    frequencies = np.asarray(segment.frequencies_hz)[mask]
    average = np.asarray(segment.average_power_linear)[mask]
    max_hold = np.asarray(segment.max_hold_power_linear)[mask]

    inside = (frequencies >= band_start_hz) & (frequencies < band_stop_hz)
    frequencies, average, max_hold = frequencies[inside], average[inside], max_hold[inside]

    cells = np.floor((frequencies - band_start_hz) / resolution_hz).astype(np.intp)
    cells = np.clip(cells, 0, num_cells - 1)  # guard float rounding at the top edge

    counts = np.bincount(cells, minlength=num_cells)
    sums = np.bincount(cells, weights=average, minlength=num_cells)
    has_data = counts > 0

    cell_average = np.full(num_cells, np.nan)
    cell_average[has_data] = sums[has_data] / counts[has_data]

    cell_max = np.full(num_cells, -np.inf)
    np.maximum.at(cell_max, cells, max_hold)
    cell_max[~has_data] = np.nan

    return cell_average, cell_max


def stitch_segments(
    segments: Sequence[SweepSegment],
    band_start_hz: float = config.BAND_START_HZ,
    band_stop_hz: float = config.BAND_STOP_HZ,
    resolution_hz: float = config.STITCH_RESOLUTION_HZ,
) -> StitchedSpectrum:
    """Combine overlapping segments into one spectrum on the band grid.

    Per grid cell, in linear power:
        average  = mean of the cell averages of all segments covering it
        max hold = max of the cell maxima of all segments covering it

    Raises:
        ValueError: if there are no segments or a segment is inconsistent.
    """
    if not segments:
        raise ValueError("No segments to stitch.")

    frequencies = make_band_grid(band_start_hz, band_stop_hz, resolution_hz)
    average_sum = np.zeros(frequencies.size)
    max_hold = np.full(frequencies.size, -np.inf)
    segment_count = np.zeros(frequencies.size, dtype=np.int64)

    for segment in segments:
        cell_average, cell_max = segment_to_grid(
            segment, band_start_hz, band_stop_hz, resolution_hz
        )
        has_data = ~np.isnan(cell_average)
        average_sum[has_data] += cell_average[has_data]
        max_hold[has_data] = np.maximum(max_hold[has_data], cell_max[has_data])
        segment_count[has_data] += 1

    covered = segment_count > 0
    average = np.full(frequencies.size, np.nan)
    average[covered] = average_sum[covered] / segment_count[covered]
    max_hold[~covered] = np.nan

    return StitchedSpectrum(
        frequencies_hz=frequencies,
        average_power_linear=average,
        max_hold_power_linear=max_hold,
        segment_count=segment_count,
        resolution_hz=float(resolution_hz),
    )


def plan_sweep_coverage(
    center_frequencies_hz: Sequence[int] = config.SWEEP_CENTER_FREQUENCIES_HZ,
    sample_rate_hz: int = config.SAMPLE_RATE,
    fft_size: int = config.RX_BUFFER_SIZE,
    band_start_hz: float = config.BAND_START_HZ,
    band_stop_hz: float = config.BAND_STOP_HZ,
    resolution_hz: float = config.STITCH_RESOLUTION_HZ,
    dc_exclusion_hz: float = config.DC_EXCLUSION_HZ,
    edge_exclusion_hz: float = config.EDGE_EXCLUSION_HZ,
) -> StitchedSpectrum:
    """Stitch unit-power placeholder segments to check a center list offline.

    Uses exactly the same masks and stitching as a real sweep, so its
    coverage is the coverage a real sweep will reach. No hardware access.
    """
    segments = []
    for center in center_frequencies_hz:
        frequencies = frequency_axis(center, sample_rate_hz, fft_size)
        mask = get_valid_spectrum_mask(
            frequencies, center, sample_rate_hz, dc_exclusion_hz, edge_exclusion_hz
        )
        ones = np.ones(fft_size)
        segments.append(SweepSegment(int(center), frequencies, ones, ones, mask))
    return stitch_segments(segments, band_start_hz, band_stop_hz, resolution_hz)


# -- hardware sweep ----------------------------------------------------------


class SpectrumSweeper:
    """Steps a connected, configured :class:`PlutoReceiver` across the band.

    Typical use:
        sweeper = SpectrumSweeper(receiver, log=print)
        sweeper.check_fixed_gain()
        segments = sweeper.run()
        stitched = stitch_segments(segments)
    """

    def __init__(
        self,
        receiver: PlutoReceiver,
        center_frequencies_hz: Sequence[int] = config.SWEEP_CENTER_FREQUENCIES_HZ,
        num_captures: int = config.NUM_AVERAGES,
        warmup_buffers: int = config.SWEEP_WARMUP_BUFFERS,
        settle_time_s: float = config.SWEEP_SETTLE_TIME_S,
        dc_exclusion_hz: float = config.DC_EXCLUSION_HZ,
        edge_exclusion_hz: float = config.EDGE_EXCLUSION_HZ,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        if not center_frequencies_hz:
            raise ValueError("At least one sweep center frequency is required.")
        if num_captures < 1:
            raise ValueError(f"num_captures must be >= 1, got {num_captures}.")
        if warmup_buffers < 0:
            raise ValueError(f"warmup_buffers must be >= 0, got {warmup_buffers}.")

        self.receiver = receiver
        self.center_frequencies_hz = tuple(int(f) for f in center_frequencies_hz)
        self.num_captures = int(num_captures)
        self.warmup_buffers = int(warmup_buffers)
        self.settle_time_s = float(settle_time_s)
        self.dc_exclusion_hz = dc_exclusion_hz
        self.edge_exclusion_hz = edge_exclusion_hz
        self.log = log

        self._reference_gain: Optional[tuple[str, float]] = None

    def check_fixed_gain(self) -> tuple[str, float]:
        """Verify on the hardware that gain is manual and as configured.

        The values read here become the reference that every segment is
        checked against.

        Returns:
            ``(gain_control_mode, gain_db)`` as reported by the Pluto.

        Raises:
            SweepError: if AGC is active or the gain differs from the config.
        """
        mode, gain_db = self._read_gain(center_frequency_hz=None)
        if mode != "manual":
            raise SweepError(
                f"RX gain control mode is '{mode}', but the sweep requires "
                "'manual' so that all segments use the same gain. "
                "Set RX_GAIN_CONTROL_MODE = \"manual\" in src/config.py."
            )
        if abs(gain_db - self.receiver.gain_db) > GAIN_TOLERANCE_DB:
            raise SweepError(
                f"Pluto reports RX gain {gain_db:g} dB, but {self.receiver.gain_db} dB "
                "was configured."
            )
        self._reference_gain = (mode, gain_db)
        return mode, gain_db

    def run(self) -> list[SweepSegment]:
        """Capture one segment per center frequency, in order.

        Raises:
            SweepError: on the first segment that fails. No partial list is
                returned.
        """
        if self._reference_gain is None:
            self.check_fixed_gain()

        segments = []
        total = len(self.center_frequencies_hz)
        for index, center in enumerate(self.center_frequencies_hz, start=1):
            self.log(f"\n[{index}/{total}] Tuning to {center / HZ_PER_MHZ:.1f} MHz")
            segments.append(self.capture_segment(center))
        return segments

    def capture_segment(self, center_frequency_hz: int) -> SweepSegment:
        """Tune, settle, discard warm-up buffers, capture and process one segment.

        Raises:
            SweepError: naming the center frequency if any step fails.
        """
        center = int(center_frequency_hz)
        label = f"{center / HZ_PER_MHZ:.1f} MHz"

        try:
            self.receiver.set_center_freq(center)
        except PlutoConnectionError as exc:
            raise SweepError(f"Tuning to {label} failed. {exc}", center) from exc
        time.sleep(self.settle_time_s)

        self.log("      Warming up...")
        for index in range(1, self.warmup_buffers + 1):
            self._receive(center, f"warm-up buffer {index}/{self.warmup_buffers}")

        captures = []
        for index in range(1, self.num_captures + 1):
            self.log(f"      Capture {index}/{self.num_captures}")
            captures.append(self._receive(center, f"capture {index}/{self.num_captures}"))

        self._verify_gain_unchanged(center)

        try:
            average, max_hold = compute_average_and_max_hold_linear(captures)
            frequencies = frequency_axis(center, self.receiver.sample_rate, average.size)
            valid_mask = get_valid_spectrum_mask(
                frequencies,
                center,
                self.receiver.sample_rate,
                self.dc_exclusion_hz,
                self.edge_exclusion_hz,
            )
        except ValueError as exc:
            raise SweepError(f"Segment at {label}: {exc}", center) from exc

        return SweepSegment(
            center_frequency_hz=center,
            frequencies_hz=frequencies,
            average_power_linear=average,
            max_hold_power_linear=max_hold,
            valid_mask=valid_mask,
        )

    # -- internals ----------------------------------------------------------

    def _receive(self, center: int, what: str) -> np.ndarray:
        try:
            return self.receiver.receive_samples()
        except PlutoConnectionError as exc:
            raise SweepError(
                f"{center / HZ_PER_MHZ:.1f} MHz: {what} failed. {exc}", center
            ) from exc

    def _read_gain(self, center_frequency_hz: Optional[int]) -> tuple[str, float]:
        try:
            return self.receiver.read_gain()
        except PlutoConnectionError as exc:
            where = (
                f"At {center_frequency_hz / HZ_PER_MHZ:.1f} MHz: "
                if center_frequency_hz is not None
                else ""
            )
            raise SweepError(f"{where}{exc}", center_frequency_hz) from exc

    def _verify_gain_unchanged(self, center: int) -> None:
        """Raise if the hardware gain differs from the sweep's reference."""
        if self._reference_gain is None:
            return
        mode, gain_db = self._read_gain(center)
        reference_mode, reference_gain_db = self._reference_gain
        if mode != reference_mode or abs(gain_db - reference_gain_db) > GAIN_TOLERANCE_DB:
            raise SweepError(
                f"RX gain changed during the segment at {center / HZ_PER_MHZ:.1f} MHz: "
                f"now {mode} / {gain_db:g} dB, sweep started with "
                f"{reference_mode} / {reference_gain_db:g} dB.",
                center,
            )
