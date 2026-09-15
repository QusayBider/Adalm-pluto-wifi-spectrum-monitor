"""Steps 4-5: Wi-Fi channel mapping, energy occupancy and center candidates.

Works offline on the spectrum saved by the last full sweep, so the detector
can be tuned without capturing RF data again. Run from the project root:
    python -m scripts.test_channel_detection

Input (read only, never modified):
    data/processed/full_band_spectrum.npz     from python -m scripts.test_full_sweep
Outputs:
    data/processed/channel_occupancy.csv
    results/figures/full_band_channels.png

Energy-based sensing only: neither "energy occupied" nor "center candidate"
means that an 802.11 frame was decoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Also allow "python scripts/test_channel_detection.py" from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.channel_detector import (
    STATUS_NO_DATA,
    STATUS_OCCUPIED,
    ChannelOccupancyReport,
    detect_channel_occupancy,
    format_channel_table,
    save_occupancy_csv,
)
from src.utils import save_figure, show_figures

SPECTRUM_PATH = PROJECT_ROOT / "data" / "processed" / "full_band_spectrum.npz"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "channel_occupancy.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "full_band_channels.png"

REQUIRED_KEYS = ("frequencies_hz", "average_power_linear", "average_power_db")
HZ_PER_MHZ = 1e6

STATUS_COLORS = {STATUS_OCCUPIED: "tab:red", "FREE": "tab:green", STATUS_NO_DATA: "tab:grey"}
CANDIDATE_COLOR = "tab:purple"


def load_spectrum(path: Path) -> dict[str, np.ndarray]:
    """Load the stitched spectrum saved by test_full_sweep.

    Raises:
        FileNotFoundError / KeyError: with a hint on how to create the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(PROJECT_ROOT)} not found. "
            "Run `python -m scripts.test_full_sweep` first."
        )
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(
                f"{path.relative_to(PROJECT_ROOT)} lacks {missing}. "
                "Re-run `python -m scripts.test_full_sweep`."
            )
        return {key: data[key] for key in data.files}


def plot_channels(
    frequencies_hz: np.ndarray,
    power_db: np.ndarray,
    report: ChannelOccupancyReport,
) -> plt.Figure:
    """Spectrum with noise-floor references, channel markers and a status strip.

    Strip box color = energy occupancy of the channel window. Center
    candidates get a solid purple marker, a star, a label and a thin bracket
    showing their nominal 20 MHz extent (no area shading).
    """
    figure, (axes, strip) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [6, 1]},
    )
    frequencies_mhz = frequencies_hz / HZ_PER_MHZ
    candidates = set(report.center_candidates)
    top = axes.get_xaxis_transform()  # x in data, y in axes fraction

    for result in report.results:
        center_mhz = result.center_frequency_hz / HZ_PER_MHZ
        is_candidate = result.channel_number in candidates
        axes.axvline(center_mhz, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
        strip.text(
            center_mhz, 0.5, str(result.channel_number),
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": STATUS_COLORS[result.status],
                  "edgecolor": CANDIDATE_COLOR if is_candidate else "none",
                  "linewidth": 2.5},
        )
        if not is_candidate:
            continue
        half_mhz = config.WIFI_CHANNEL_BANDWIDTH_HZ / 2 / HZ_PER_MHZ
        axes.axvline(center_mhz, color=CANDIDATE_COLOR, linewidth=1.6, alpha=0.6, zorder=1)
        axes.annotate(
            "", xy=(center_mhz - half_mhz, 0.9), xytext=(center_mhz + half_mhz, 0.9),
            xycoords=top, textcoords=top,
            arrowprops={"arrowstyle": "|-|", "color": CANDIDATE_COLOR, "linewidth": 1.2},
        )
        axes.plot(center_mhz, 0.9, marker="*", markersize=13, color=CANDIDATE_COLOR,
                  transform=top, clip_on=False)
        axes.text(center_mhz, 0.94, f"Candidate CH{result.channel_number}", transform=top,
                  ha="center", va="bottom", fontsize=9, color=CANDIDATE_COLOR,
                  fontweight="bold")

    axes.plot(frequencies_mhz, power_db, linewidth=0.6, color="tab:blue",
              label="Stitched average spectrum")
    axes.axhline(report.noise_floor_db, color="black", linestyle="--", linewidth=1.0,
                 label=f"Noise floor {report.noise_floor_db:.1f} dB")
    axes.axhline(report.active_threshold_db, color="tab:orange", linestyle="--", linewidth=1.0,
                 label=f"Active-bin threshold (+{report.settings.active_bin_margin_db:g} dB)")

    handles, _ = axes.get_legend_handles_labels()
    handles += [
        Patch(color=STATUS_COLORS[STATUS_OCCUPIED], label="Energy in channel window"),
        Patch(color=STATUS_COLORS["FREE"], label="No significant energy"),
        Line2D([], [], color=CANDIDATE_COLOR, marker="*", markersize=10, linewidth=1.6,
               label="Center candidate (±10 MHz)"),
    ]
    axes.legend(handles=handles, loc="center right", fontsize=9)
    axes.set_title("2.4 GHz Wi-Fi Channel Occupancy")
    axes.set_ylabel("Relative Power (dB)")
    axes.grid(True, alpha=0.3)

    strip.set_xlim(config.BAND_START_HZ / HZ_PER_MHZ, config.BAND_STOP_HZ / HZ_PER_MHZ)
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.set_ylabel("Channel", rotation=0, ha="right", va="center")
    strip.set_xlabel("Frequency (MHz)")
    for side in ("left", "right", "top"):
        strip.spines[side].set_visible(False)
    return figure


def main() -> int:
    """Analyse the saved full-band spectrum; write the table, CSV and figure.

    Returns:
        Process exit code (0 on success, 1 if the spectrum is missing/invalid).
    """
    try:
        spectrum = load_spectrum(SPECTRUM_PATH)
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    frequencies_hz = spectrum["frequencies_hz"]
    try:
        report = detect_channel_occupancy(
            frequencies_hz,
            spectrum["average_power_linear"],
            spectrum["average_power_db"],
        )
    except ValueError as exc:
        print(f"Error: channel analysis failed - {exc}")
        return 1

    settings = report.settings
    print("Energy-based spectrum sensing - no 802.11 frames are decoded.")
    print(
        "  Energy:    noise floor = P"
        f"{settings.noise_floor_percentile:g}, active bin >= floor + "
        f"{settings.active_bin_margin_db:g} dB; window occupied if median excess >= "
        f"{settings.channel_power_margin_db:g} dB or active bins >= "
        f"{100 * settings.min_active_bin_fraction:g} %"
    )
    print(
        f"  Candidate: core ±{settings.core_half_bandwidth_hz / HZ_PER_MHZ:g} MHz, "
        "score = max(core excess, 0) × core active fraction, "
        f"local max over ±{settings.candidate_neighbor_channels} ch, "
        f"score >= {settings.min_center_score:g}, "
        f"separation >= {settings.min_candidate_separation_hz / HZ_PER_MHZ:g} MHz"
    )
    print()
    print(f"Estimated Noise Floor: {report.noise_floor_db:.2f} dB")
    print()
    print(format_channel_table(report))
    print()
    print("Energy-active channel windows:")
    print("  (significant RF energy inside the standard 20 MHz channel window)")
    print(f"  {report.energy_occupied_channels}")
    print("Likely Wi-Fi channel center candidates:")
    print("  (broadband energy most consistent with a signal centered on these channels)")
    print(f"  {report.center_candidates}")
    print()

    sweep_timestamp = str(spectrum.get("timestamp_utc", ""))
    figure = plot_channels(frequencies_hz, spectrum["average_power_db"], report)
    try:
        save_occupancy_csv(CSV_PATH, report, sweep_timestamp)
        print("Channel results saved to:")
        print(CSV_PATH.relative_to(PROJECT_ROOT))
        save_figure(figure, FIGURE_PATH)
    except OSError as exc:
        print(f"Error: could not save results - {exc}")
        return 1

    print()
    if sweep_timestamp:
        print(f"Sweep timestamp (UTC): {sweep_timestamp}")
    print(f"Spectrum bins loaded: {frequencies_hz.size}")
    print(
        f"Frequency range: {frequencies_hz[0] / HZ_PER_MHZ:.1f}–"
        f"{frequencies_hz[-1] / HZ_PER_MHZ:.1f} MHz"
    )
    print(f"Estimated noise floor: {report.noise_floor_db:.2f} dB")
    print(f"Channels analyzed: {len(report.results)}")
    print(f"Energy-occupied channel windows: {report.energy_occupied_channels}")
    print(f"Energy-free channel windows: {report.free_channels}")
    print(f"Center candidates: {report.center_candidates}")
    if report.no_data_channels:
        print(f"Channels without data: {report.no_data_channels}")

    show_figures()
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
