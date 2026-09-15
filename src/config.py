"""Central configuration of the ADALM-PLUTO 2.4 GHz Wi-Fi spectrum monitor.

Conventions:
    * Frequencies in Hz, rates in samples/second, powers in relative dB.
    * Values handed to the SDR are ``int``: libiio attributes are integers.
    * Detection thresholds are relative to the estimated noise floor; the
      receiver is not calibrated, so there are no absolute (dBm) levels.

All values below were validated on the hardware. Thresholds marked as
experimental are expected to be revisited with controlled measurements.
"""

from __future__ import annotations

# =============================================================================
# Pluto connection
# =============================================================================
# libiio context URI. Run `iio_info -S usb` and copy the URI in square
# brackets from the Pluto line, e.g. "[usb:1.2.5]". "usb:" selects the only
# USB Pluto automatically; the bus.address numbers can change whenever
# usbipd re-attaches the device.
PLUTO_URI = "usb:1.2.5"

# =============================================================================
# Receiver
# =============================================================================
CENTER_FREQUENCY: int = int(2.437e9)  # Single-channel tests (Wi-Fi channel 6)
SAMPLE_RATE: int = int(20e6)          # 20 MS/s -> 20 MHz observed per capture
RF_BANDWIDTH: int = int(20e6)         # Analog RX filter, matched to sample rate
RX_BUFFER_SIZE: int = 16384           # Complex samples per rx() call = FFT size

# Fixed manual gain keeps relative power comparable between captures and sweep
# segments; AGC would change the gain per LO frequency. The sweep refuses to
# run unless the hardware reports manual mode with this gain.
RX_GAIN_CONTROL_MODE: str = "manual"  # "manual" | "slow_attack" | "fast_attack"
RX_HARDWARE_GAIN_DB: int = 30         # 0..71 dB, used only in manual mode

# =============================================================================
# Spectrum
# =============================================================================
# Added before log10() so zero-power bins cannot produce -inf.
POWER_EPSILON: float = 1e-20

# RX buffers combined (linear power) into one average / max-hold spectrum.
NUM_AVERAGES: int = 20

# Buffers discarded after tuning in the single-channel test (test_spectrum).
NUM_WARMUP_BUFFERS: int = 2

# Valid-bin mask: bins in these regions are excluded from stitching and
# analysis (the raw spectra are never modified).
DC_EXCLUSION_HZ: int = 200_000       # LO center +/- this (LO leakage)
EDGE_EXCLUSION_HZ: int = 1_000_000   # from each capture edge (filter roll-off)

# =============================================================================
# Sweep
# =============================================================================
BAND_START_HZ: int = int(2.400e9)
BAND_STOP_HZ: int = int(2.500e9)

# LO centers. Each capture is valid over center -9 ... +9 MHz except a
# +/-0.2 MHz hole at the center, which only a capture 0.4 ... 8.8 MHz away can
# fill. Centers are therefore placed in pairs 8 MHz apart, 25 MHz between
# pairs, giving 100 % coverage with 1 MHz overlap between pairs and 0.5 MHz
# margin beyond both band edges:
#   2408.5 / 2416.5 -> 2399.5 ... 2425.5    2458.5 / 2466.5 -> 2449.5 ... 2475.5
#   2433.5 / 2441.5 -> 2424.5 ... 2450.5    2483.5 / 2491.5 -> 2474.5 ... 2500.5
SWEEP_CENTER_FREQUENCIES_HZ: tuple[int, ...] = (
    2_408_500_000,
    2_416_500_000,
    2_433_500_000,
    2_441_500_000,
    2_458_500_000,
    2_466_500_000,
    2_483_500_000,
    2_491_500_000,
)

# After every LO change: wait, then discard this many buffers before measuring.
SWEEP_SETTLE_TIME_S: float = 0.01
SWEEP_WARMUP_BUFFERS: int = 2

# Cell width of the common 2400-2500 MHz grid. Each cell holds the mean linear
# power of the native FFT bins (1220.7 Hz) inside it (>= 4 bins per cell).
STITCH_RESOLUTION_HZ: int = 5_000

# =============================================================================
# Occupancy detection (energy per standard Wi-Fi channel window)
# =============================================================================
# Channel centers (2407 MHz + 5 MHz * channel). Channels are 5 MHz apart but
# about 20 MHz wide, so neighbouring windows overlap.
WIFI_CHANNELS_HZ: dict[int, int] = {
    1: 2_412_000_000,
    2: 2_417_000_000,
    3: 2_422_000_000,
    4: 2_427_000_000,
    5: 2_432_000_000,
    6: 2_437_000_000,
    7: 2_442_000_000,
    8: 2_447_000_000,
    9: 2_452_000_000,
    10: 2_457_000_000,
    11: 2_462_000_000,
    12: 2_467_000_000,
    13: 2_472_000_000,
}
WIFI_CHANNEL_BANDWIDTH_HZ: int = 20_000_000  # window = center +/- half of this

# Experimental values. Noise floor = this percentile of all covered bins of
# the stitched average spectrum (robust while < ~80 % of the band is busy).
NOISE_FLOOR_PERCENTILE: float = 20.0

# A bin is "active" if it is at least this far above the noise floor.
ACTIVE_BIN_MARGIN_DB: float = 6.0

# ENERGY OCCUPIED if  median window power - noise floor >= CHANNEL_POWER_MARGIN_DB
#                 or  active-bin fraction >= MIN_ACTIVE_BIN_FRACTION.
# Peak power is reported but never decides.
CHANNEL_POWER_MARGIN_DB: float = 5.0
MIN_ACTIVE_BIN_FRACTION: float = 0.20

# =============================================================================
# Candidate detection (which channel center best explains broadband energy)
# =============================================================================
# Experimental values. Core region analysed per channel: center +/- this.
CORE_HALF_BANDWIDTH_HZ: int = 8_000_000

# center_score = max(core median - noise floor, 0 dB) * core active fraction.
# 2.5 corresponds e.g. to a core 5 dB above the floor with half its bins active.
MIN_CENTER_SCORE: float = 2.5

# A candidate's score must be >= that of all channels within this many numbers.
CANDIDATE_NEIGHBOR_CHANNELS: int = 1

# Non-maximum suppression: drop a weaker candidate closer than this to a
# stronger accepted one (15 MHz = 3 channel numbers).
MIN_CANDIDATE_SEPARATION_HZ: int = 15_000_000

# =============================================================================
# Temporal filtering (stable candidates over recent sweeps)
# =============================================================================
# Center scores are averaged over this many most recent sweeps, then the
# candidate rules above are applied to the averages.
TEMPORAL_WINDOW_SWEEPS: int = 5

# A stable candidate's own score must reach MIN_CENTER_SCORE in at least this
# many sweeps of the window (all available sweeps while fewer exist).
TEMPORAL_MIN_SUPPORT: int = 3

# =============================================================================
# Live monitoring
# =============================================================================
MONITOR_NUM_SWEEPS: int = 30

# Minimum time from one sweep start to the next; 0 = back-to-back, no sleep.
MONITOR_INTERVAL_SECONDS: float = 0.0

# Only used for the startup time estimate (measured full-band sweep time).
EXPECTED_SWEEP_SECONDS: float = 1.1

# =============================================================================
# Plotting
# =============================================================================
# Most recent sweeps shown in the live heatmap (all sweeps are still saved).
HEATMAP_HISTORY_LENGTH: int = 30

# Heatmap color scale (display only; stored data is never clipped).
HEATMAP_DB_MIN: float = -45.0
HEATMAP_DB_MAX: float = -10.0

# Heatmap image column width: grid cells are averaged in linear power into
# columns of this width for display only. Multiple of STITCH_RESOLUTION_HZ.
HEATMAP_DISPLAY_BIN_HZ: int = 50_000
