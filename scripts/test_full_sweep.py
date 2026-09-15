"""Step 3: sweep 2400-2500 MHz, stitch the segments, save plots and data.

Run from the project root:
    python -m scripts.test_full_sweep

Outputs:
    results/figures/full_band_spectrum.png          stitched average
    results/figures/full_band_average_maxhold.png  stitched average + max hold
    results/figures/full_band_segments.png         valid part of every segment (debug)
    data/processed/full_band_spectrum.npz          numerical result + metadata
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Also allow "python scripts/test_full_sweep.py" from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pluto_receiver import PlutoConnectionError, PlutoReceiver
from src.spectrum import power_to_db
from src.sweep import (
    SpectrumSweeper,
    StitchedSpectrum,
    SweepError,
    SweepSegment,
    plan_sweep_coverage,
    stitch_segments,
)
from src.utils import save_figure, show_figures

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FULL_BAND_FIGURE_PATH = FIGURES_DIR / "full_band_spectrum.png"
AVERAGE_MAXHOLD_FIGURE_PATH = FIGURES_DIR / "full_band_average_maxhold.png"
SEGMENTS_FIGURE_PATH = FIGURES_DIR / "full_band_segments.png"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "full_band_spectrum.npz"

HZ_PER_MHZ = 1e6
BAND_START_MHZ = config.BAND_START_HZ / HZ_PER_MHZ
BAND_STOP_MHZ = config.BAND_STOP_HZ / HZ_PER_MHZ
MAX_LISTED_GAPS = 10


# -- plotting ----------------------------------------------------------------


def style_band_axes(axes: plt.Axes, title: str) -> None:
    """Common title, axis labels, 2400-2500 MHz limits and grid."""
    axes.set_title(title)
    axes.set_xlabel("Frequency (MHz)")
    axes.set_ylabel("Relative Power (dB)")
    axes.set_xlim(BAND_START_MHZ, BAND_STOP_MHZ)
    axes.grid(True, alpha=0.3)


def mark_sweep_centers(axes: plt.Axes, centers_hz: Sequence[int]) -> None:
    """Very light vertical lines at the LO centers."""
    for center in centers_hz:
        axes.axvline(center / HZ_PER_MHZ, color="grey", linestyle=":", linewidth=0.8, alpha=0.4)


def plot_full_band(stitched: StitchedSpectrum, centers_hz: Sequence[int]) -> plt.Figure:
    """Stitched average spectrum over the full band."""
    figure, axes = plt.subplots(figsize=(12, 5))
    mark_sweep_centers(axes, centers_hz)
    axes.plot(
        stitched.frequencies_hz / HZ_PER_MHZ,
        stitched.average_power_db,
        linewidth=0.7,
        label=f"Stitched average ({config.NUM_AVERAGES} captures per center)",
    )
    style_band_axes(axes, "ADALM-PLUTO Full 2.4 GHz Band Spectrum")
    axes.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_average_maxhold(stitched: StitchedSpectrum, centers_hz: Sequence[int]) -> plt.Figure:
    """Stitched average and stitched max hold on one axis."""
    frequencies_mhz = stitched.frequencies_hz / HZ_PER_MHZ
    figure, axes = plt.subplots(figsize=(12, 5))
    mark_sweep_centers(axes, centers_hz)
    axes.plot(frequencies_mhz, stitched.average_power_db, linewidth=0.7, label="Stitched average")
    axes.plot(frequencies_mhz, stitched.max_hold_power_db, linewidth=0.7, label="Stitched max hold")
    style_band_axes(axes, "ADALM-PLUTO Full 2.4 GHz Band Spectrum")
    axes.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_segments(segments: Sequence[SweepSegment]) -> plt.Figure:
    """Valid bins of every segment in its own color; excluded bins are gaps."""
    figure, axes = plt.subplots(figsize=(13, 6))
    for segment in segments:
        power_db = np.where(
            segment.valid_mask, power_to_db(segment.average_power_linear), np.nan
        )
        axes.plot(
            segment.frequencies_hz / HZ_PER_MHZ,
            power_db,
            linewidth=0.5,
            alpha=0.7,
            label=f"{segment.center_frequency_hz / HZ_PER_MHZ:.1f} MHz",
        )
    style_band_axes(axes, "ADALM-PLUTO Sweep Segments - Valid Bins Before Stitching")
    legend = axes.legend(
        title="LO center", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.0)
        handle.set_alpha(1.0)
    figure.tight_layout()
    return figure


# -- saving ------------------------------------------------------------------


def save_numerical_results(
    path: Path,
    stitched: StitchedSpectrum,
    receiver: PlutoReceiver,
    sweeper: SpectrumSweeper,
    gain_mode: str,
    gain_db: float,
) -> None:
    """Save the stitched spectrum (linear + dB) and its capture metadata as .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frequencies_hz=stitched.frequencies_hz,
        average_power_linear=stitched.average_power_linear,
        average_power_db=stitched.average_power_db,
        max_hold_power_linear=stitched.max_hold_power_linear,
        max_hold_power_db=stitched.max_hold_power_db,
        segment_count=stitched.segment_count,
        sample_rate=receiver.sample_rate,
        rf_bandwidth=receiver.rf_bandwidth,
        fft_size=receiver.buffer_size,
        sweep_centers_hz=np.array(sweeper.center_frequencies_hz, dtype=np.int64),
        rx_gain_db=gain_db,
        rx_gain_control_mode=gain_mode,
        captures_per_center=sweeper.num_captures,
        warmup_buffers=sweeper.warmup_buffers,
        stitch_resolution_hz=stitched.resolution_hz,
        dc_exclusion_hz=sweeper.dc_exclusion_hz,
        edge_exclusion_hz=sweeper.edge_exclusion_hz,
        band_start_hz=config.BAND_START_HZ,
        band_stop_hz=config.BAND_STOP_HZ,
        coverage_percent=stitched.coverage_percent,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    print("Data saved to:")
    print(path.relative_to(PROJECT_ROOT))


def print_coverage(stitched: StitchedSpectrum, prefix: str) -> None:
    """Print coverage percentage, uncovered bin count and the first gaps."""
    print(f"{prefix}: {stitched.coverage_percent:.2f} %")
    print(f"Uncovered frequency bins: {stitched.uncovered_bins} / {stitched.segment_count.size}")
    gaps = stitched.uncovered_ranges_hz()
    for start_hz, stop_hz in gaps[:MAX_LISTED_GAPS]:
        print(f"  gap: {start_hz / HZ_PER_MHZ:.3f} - {stop_hz / HZ_PER_MHZ:.3f} MHz")
    if len(gaps) > MAX_LISTED_GAPS:
        print(f"  ... and {len(gaps) - MAX_LISTED_GAPS} more gaps")


# -- main --------------------------------------------------------------------


def _progress_without_captures(message: str) -> None:
    """Sweep log that shows the LO steps but not every warm-up/capture line."""
    if not message.strip().startswith(("Capture", "Warming up")):
        print(message)


def main(verbose: bool = True) -> int:
    """Run one full-band sweep and save figures and full_band_spectrum.npz.

    Args:
        verbose: Print every warm-up and capture step (True) or only the LO
            steps (False, used by main.py).

    Returns:
        Process exit code: 0 on success with full coverage, 1 otherwise.
    """
    centers = config.SWEEP_CENTER_FREQUENCIES_HZ

    receiver = PlutoReceiver(uri=config.PLUTO_URI, center_freq=centers[0])
    log = print if verbose else _progress_without_captures
    sweeper = SpectrumSweeper(receiver, centers, log=log)

    try:
        receiver.connect()
        receiver.configure()
        gain_mode, gain_db = sweeper.check_fixed_gain()

        print("Connected to ADALM-PLUTO")
        print(f"Sweep Range: {BAND_START_MHZ:.0f}–{BAND_STOP_MHZ:.0f} MHz")
        print(f"Sample Rate: {receiver.sample_rate / HZ_PER_MHZ:.1f} MSPS")
        print(f"FFT Size: {receiver.buffer_size}")
        print(f"Captures per center: {sweeper.num_captures}")
        print(f"RX Gain Mode: {gain_mode}")
        print(f"RX Gain: {gain_db:g} dB")
        print(f"Sweep Centers: {len(centers)}")

        planned = plan_sweep_coverage(
            centers,
            sample_rate_hz=receiver.sample_rate,
            fft_size=receiver.buffer_size,
        )
        print(f"Planned coverage: {planned.coverage_percent:.2f} %")
        if planned.uncovered_bins:
            print(
                "WARNING: SWEEP_CENTER_FREQUENCIES_HZ leaves gaps; they will be NaN "
                "in the result."
            )

        start_time = time.monotonic()
        segments = sweeper.run()
        print(f"\nSweep time: {time.monotonic() - start_time:.1f} s")
    except SweepError as exc:
        print(f"\nSweep failed: {exc}")
        return 1
    except PlutoConnectionError as exc:
        print(f"Error: {exc}")
        print("Run `python -m scripts.test_pluto` first to debug the connection.")
        return 1
    finally:
        receiver.close()

    print("Stitching spectrum...")
    try:
        stitched = stitch_segments(segments)
    except ValueError as exc:
        print(f"Stitching failed: {exc}")
        return 1
    print_coverage(stitched, "Full-band coverage")

    figures = [
        (FULL_BAND_FIGURE_PATH, plot_full_band(stitched, centers)),
        (AVERAGE_MAXHOLD_FIGURE_PATH, plot_average_maxhold(stitched, centers)),
        (SEGMENTS_FIGURE_PATH, plot_segments(segments)),
    ]
    try:
        save_numerical_results(DATA_PATH, stitched, receiver, sweeper, gain_mode, gain_db)
        for path, figure in figures:
            save_figure(figure, path)
    except OSError as exc:
        print(f"Error: could not save results - {exc}")
        return 1

    if stitched.uncovered_bins:
        print(
            "Sweep finished, but full-band coverage is incomplete. "
            "Uncovered bins are NaN in the saved data."
        )
        exit_code = 1
    else:
        print("Sweep completed successfully.")
        exit_code = 0

    show_figures()
    plt.close("all")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
