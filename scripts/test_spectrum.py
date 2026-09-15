"""Step 2: capture Wi-Fi channel 6 and plot raw, average and max-hold spectra.

Run from the project root:
    python -m scripts.test_spectrum

Outputs (results/figures/):
    channel6_spectrum.png          single raw capture
    channel6_average_maxhold.png   average vs. max hold over NUM_AVERAGES captures
    channel6_comparison.png        single raw capture vs. average
    channel6_clean_analysis.png    average vs. max hold, excluded bins shaded

All PNGs are written before an interactive window is attempted, so a missing
display cannot lose the results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Also allow "python scripts/test_spectrum.py" from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pluto_receiver import PlutoConnectionError, PlutoReceiver
from src.spectrum import (
    compute_average_and_max_hold,
    compute_power_spectrum,
    get_valid_spectrum_mask,
)
from src.utils import save_figure, show_figures

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SINGLE_FIGURE_PATH = FIGURES_DIR / "channel6_spectrum.png"
AVERAGE_MAXHOLD_FIGURE_PATH = FIGURES_DIR / "channel6_average_maxhold.png"
COMPARISON_FIGURE_PATH = FIGURES_DIR / "channel6_comparison.png"
CLEAN_ANALYSIS_FIGURE_PATH = FIGURES_DIR / "channel6_clean_analysis.png"

PLOT_TITLE = "ADALM-PLUTO Wi-Fi Spectrum - Channel 6"
HZ_PER_MHZ = 1e6


def capture_buffers(
    receiver: PlutoReceiver,
    num_captures: int,
    num_warmup: int,
) -> list[np.ndarray]:
    """Discard ``num_warmup`` buffers, then return ``num_captures`` IQ buffers.

    Raises:
        PlutoConnectionError: naming the capture that failed. No partial
            result is returned.
    """
    for index in range(1, num_warmup + 1):
        try:
            receiver.receive_samples()
        except PlutoConnectionError as exc:
            raise PlutoConnectionError(
                f"Warm-up buffer {index}/{num_warmup} failed. {exc}"
            ) from exc

    captures = []
    for index in range(1, num_captures + 1):
        print(f"Capture {index}/{num_captures}")
        try:
            captures.append(receiver.receive_samples())
        except PlutoConnectionError as exc:
            raise PlutoConnectionError(
                f"Capture {index}/{num_captures} failed. {exc}"
            ) from exc
    return captures


def excluded_spans(
    frequencies_hz: np.ndarray,
    valid_mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Return ``(start_hz, stop_hz)`` of every contiguous run of invalid bins."""
    invalid = np.concatenate(([0], (~valid_mask).astype(np.int8), [0]))
    changes = np.diff(invalid)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(frequencies_hz[a], frequencies_hz[b]) for a, b in zip(starts, stops)]


def plot_spectra(
    frequencies_hz: np.ndarray,
    curves: Sequence[tuple[str, np.ndarray]],
    center_frequency: int,
    valid_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Plot one or more ``(label, power_db)`` curves on a shared frequency axis.

    If ``valid_mask`` is given, the excluded bins are lightly shaded. The
    curves are always drawn unmodified.
    """
    frequencies_mhz = frequencies_hz / HZ_PER_MHZ
    center_mhz = center_frequency / HZ_PER_MHZ

    figure, axes = plt.subplots(figsize=(10, 5))
    if valid_mask is not None:
        for index, (start_hz, stop_hz) in enumerate(excluded_spans(frequencies_hz, valid_mask)):
            axes.axvspan(
                start_hz / HZ_PER_MHZ,
                stop_hz / HZ_PER_MHZ,
                color="grey",
                alpha=0.2,
                linewidth=0,
                label="Excluded from analysis" if index == 0 else None,
            )
    for label, power_db in curves:
        axes.plot(frequencies_mhz, power_db, linewidth=0.8, label=label)
    axes.axvline(
        center_mhz,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Center {center_mhz:.1f} MHz",
    )

    axes.set_title(PLOT_TITLE)
    axes.set_xlabel("Frequency (MHz)")
    axes.set_ylabel("Relative Power (dB)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper right")
    figure.tight_layout()
    return figure


def main() -> int:
    """Capture channel 6 and save single, average/max-hold and mask figures.

    Returns:
        Process exit code (0 on success).
    """
    num_captures = config.NUM_AVERAGES

    receiver = PlutoReceiver(
        uri=config.PLUTO_URI,
        center_freq=config.CENTER_FREQUENCY,
    )
    try:
        receiver.connect()
        receiver.configure()

        print("Connected to ADALM-PLUTO")
        print(f"Center Frequency: {receiver.center_freq / HZ_PER_MHZ:.1f} MHz")
        print(f"Sample Rate: {receiver.sample_rate / HZ_PER_MHZ:.1f} MSPS")
        print(f"FFT Size: {receiver.buffer_size}")
        print(f"Number of Captures: {num_captures}")

        captures = capture_buffers(receiver, num_captures, config.NUM_WARMUP_BUFFERS)
    except PlutoConnectionError as exc:
        print(f"Error: {exc}")
        print("Run `python -m scripts.test_pluto` first to debug the connection.")
        return 1
    finally:
        receiver.close()

    try:
        single = compute_power_spectrum(
            captures[0],
            center_frequency=receiver.center_freq,
            sample_rate=receiver.sample_rate,
        )
        combined = compute_average_and_max_hold(
            captures,
            center_frequency=receiver.center_freq,
            sample_rate=receiver.sample_rate,
        )
        valid_mask = get_valid_spectrum_mask(
            combined.frequencies_hz,
            center_frequency_hz=receiver.center_freq,
            sample_rate_hz=receiver.sample_rate,
        )
    except ValueError as exc:
        print(f"Error: invalid IQ data or mask settings - {exc}")
        return 1

    print("Average and Max Hold spectra calculated successfully.")
    print(f"Valid spectrum bins: {np.count_nonzero(valid_mask)} / {valid_mask.size}")
    print(f"DC exclusion: ±{config.DC_EXCLUSION_HZ / HZ_PER_MHZ:.1f} MHz")
    print(f"Edge exclusion: {config.EDGE_EXCLUSION_HZ / HZ_PER_MHZ:.1f} MHz per side")

    average_label = f"Average ({num_captures} captures)"
    average_maxhold_curves = [
        (average_label, combined.average_power_db),
        (f"Max Hold ({num_captures} captures)", combined.max_hold_power_db),
    ]
    # (path, curves, valid_mask to shade or None)
    figures = [
        (AVERAGE_MAXHOLD_FIGURE_PATH, average_maxhold_curves, None),
        (CLEAN_ANALYSIS_FIGURE_PATH, average_maxhold_curves, valid_mask),
        (
            COMPARISON_FIGURE_PATH,
            [
                ("Single capture", single.power_db),
                (average_label, combined.average_power_db),
            ],
            None,
        ),
        (SINGLE_FIGURE_PATH, [("Single capture", single.power_db)], None),
    ]

    for path, curves, shade_mask in figures:
        figure = plot_spectra(
            combined.frequencies_hz, curves, receiver.center_freq, shade_mask
        )
        try:
            save_figure(figure, path)
        except OSError as exc:
            print(f"Error: could not save figure to {path} - {exc}")
            return 1

    show_figures()
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
