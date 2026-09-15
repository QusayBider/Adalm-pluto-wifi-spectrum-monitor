"""Hardware layer: connect to the ADALM-PLUTO and return raw IQ samples.

This module deliberately contains no signal processing and no plotting. Its
only job is to own the SDR handle, apply the RX configuration and hand
complex baseband buffers to the layers above.

Dependency chain (all inside Linux / WSL2):
    adi (pyadi-iio, pip) -> iio (pylibiio, pip) -> libiio.so.0 (libiio0, apt)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config

try:
    import adi
except (ImportError, OSError, AttributeError) as exc:  # pragma: no cover
    # ImportError: pyadi-iio / pylibiio not installed in the active venv.
    # OSError / AttributeError: the `iio` bindings are installed but the
    # native libiio shared library could not be loaded.
    raise ImportError(
        "Could not import pyadi-iio (`import adi`).\n"
        "  - Is the virtual environment active?  source .venv/bin/activate\n"
        "  - Python packages:  python -m pip install -r requirements.txt\n"
        "  - Native libiio:    sudo apt install libiio-utils\n"
        f"  Original error: {type(exc).__name__}: {exc}"
    ) from exc


class PlutoConnectionError(RuntimeError):
    """Raised when the ADALM-PLUTO cannot be reached, configured or read."""


def _root_cause(exc: BaseException) -> str:
    """Return the innermost exception in a chain as ``"Type: message"``.

    pyadi-iio replaces every connection failure with a generic
    ``Exception("No device found")`` and keeps the real libiio error (e.g.
    "Permission denied") only in the exception chain.
    """
    while exc.__cause__ is not None or exc.__context__ is not None:
        exc = exc.__cause__ or exc.__context__
    return f"{type(exc).__name__}: {exc}"


class PlutoReceiver:
    """Manages the RX path of an ADALM-PLUTO.

    Typical use:
        receiver = PlutoReceiver()
        receiver.connect()
        receiver.configure()
        samples = receiver.receive_samples()
        receiver.close()

    or, equivalently, ``with PlutoReceiver() as receiver: ...`` which
    connects and configures on entry and closes on exit.
    """

    def __init__(
        self,
        uri: str = config.PLUTO_URI,
        sample_rate: int = config.SAMPLE_RATE,
        rf_bandwidth: int = config.RF_BANDWIDTH,
        center_freq: int = config.CENTER_FREQUENCY,
        buffer_size: int = config.RX_BUFFER_SIZE,
        gain_control_mode: str = config.RX_GAIN_CONTROL_MODE,
        gain_db: int = config.RX_HARDWARE_GAIN_DB,
    ) -> None:
        self.uri = uri
        self.sample_rate = int(sample_rate)
        self.rf_bandwidth = int(rf_bandwidth)
        self.center_freq = int(center_freq)
        self.buffer_size = int(buffer_size)
        self.gain_control_mode = gain_control_mode
        self.gain_db = int(gain_db)

        self._sdr: Optional["adi.Pluto"] = None

    # -- connection ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._sdr is not None

    def connect(self) -> None:
        """Open the libiio context for ``self.uri``.

        An explicit URI is used on purpose. ``adi.Pluto()`` without one scans
        for devices and then falls back to ``ip:pluto.local`` (mDNS), which is
        unreliable under WSL2 and makes failures harder to diagnose.

        Raises:
            PlutoConnectionError: if no device answers at the URI.
        """
        if self.is_connected:
            return

        try:
            self._sdr = adi.Pluto(uri=self.uri)
        except Exception as exc:
            raise PlutoConnectionError(
                f"Could not connect to ADALM-PLUTO at URI '{self.uri}'. "
                f"Underlying error: {_root_cause(exc)}"
            ) from exc

    def configure(self) -> None:
        """Push the RX settings to the hardware.

        Raises:
            PlutoConnectionError: if not connected or a value is rejected.
        """
        sdr = self._require_sdr()
        try:
            sdr.sample_rate = self.sample_rate
            sdr.rx_rf_bandwidth = self.rf_bandwidth
            sdr.rx_lo = self.center_freq
            sdr.rx_buffer_size = self.buffer_size
            sdr.gain_control_mode_chan0 = self.gain_control_mode
            if self.gain_control_mode == "manual":
                sdr.rx_hardwaregain_chan0 = self.gain_db
        except Exception as exc:
            raise PlutoConnectionError(
                f"Connected to '{self.uri}' but failed to configure the RX path. "
                f"Underlying error: {_root_cause(exc)}"
            ) from exc

    def close(self) -> None:
        """Release the device. Safe to call more than once.

        pyadi-iio has no explicit ``close()``. The resources it holds are the
        RX buffer, freed by ``rx_destroy_buffer()``, and the libiio context,
        which is destroyed when the ``adi.Pluto`` object is garbage collected.
        Dropping our only reference therefore closes the USB connection.
        """
        if self._sdr is None:
            return
        try:
            self._sdr.rx_destroy_buffer()
        except Exception:
            pass  # No buffer was created yet, or the device already vanished.
        finally:
            self._sdr = None

    # -- runtime control ----------------------------------------------------

    def set_center_freq(self, center_freq: int) -> None:
        """Retune the RX local oscillator.

        The RX buffer is released after tuning, so the next read creates a
        fresh buffer and cannot return samples queued before the LO change.

        Raises:
            PlutoConnectionError: if the device rejects the frequency.
        """
        self.center_freq = int(center_freq)
        if not self.is_connected:
            return
        sdr = self._require_sdr()
        try:
            sdr.rx_lo = self.center_freq
        except Exception as exc:
            raise PlutoConnectionError(
                f"Failed to tune RX LO to {self.center_freq / 1e6:.3f} MHz. "
                f"Underlying error: {_root_cause(exc)}"
            ) from exc
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass  # No buffer existed yet.

    def read_gain(self) -> tuple[str, float]:
        """Read the gain control mode and RX gain (dB) back from the hardware.

        Raises:
            PlutoConnectionError: if not connected or the read fails.
        """
        sdr = self._require_sdr()
        try:
            return str(sdr.gain_control_mode_chan0), float(sdr.rx_hardwaregain_chan0)
        except Exception as exc:
            raise PlutoConnectionError(
                f"Failed to read RX gain settings. Underlying error: {_root_cause(exc)}"
            ) from exc

    # -- data ---------------------------------------------------------------

    def receive_samples(self) -> np.ndarray:
        """Return one buffer of complex IQ samples.

        Returns:
            1-D complex array of length ``buffer_size``.

        Raises:
            PlutoConnectionError: if the read fails or returns no usable data.
        """
        sdr = self._require_sdr()
        try:
            samples = sdr.rx()
        except Exception as exc:
            raise PlutoConnectionError(
                "Failed to read IQ samples from the ADALM-PLUTO. "
                f"Underlying error: {_root_cause(exc)}"
            ) from exc

        if samples is None:
            raise PlutoConnectionError("The ADALM-PLUTO returned no samples.")

        samples = np.asarray(samples)
        if samples.ndim != 1 or not np.iscomplexobj(samples):
            raise PlutoConnectionError(
                "Expected a 1-D complex IQ buffer, got "
                f"shape {samples.shape} with dtype {samples.dtype}."
            )
        if samples.size != self.buffer_size:
            raise PlutoConnectionError(
                f"Expected {self.buffer_size} samples, received {samples.size}."
            )
        return samples

    # -- internals ----------------------------------------------------------

    def _require_sdr(self) -> "adi.Pluto":
        if self._sdr is None:
            raise PlutoConnectionError("Not connected. Call connect() first.")
        return self._sdr

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "PlutoReceiver":
        self.connect()
        try:
            self.configure()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PlutoReceiver(uri={self.uri!r}, center_freq={self.center_freq}, "
            f"sample_rate={self.sample_rate}, buffer_size={self.buffer_size}, "
            f"connected={self.is_connected})"
        )
