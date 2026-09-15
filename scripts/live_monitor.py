"""Steps 6-7: repeated full-band monitoring and time-frequency heatmap.

Run from the project root:
    python -m scripts.live_monitor

Every iteration runs the existing SpectrumSweeper, stitches the segments,
runs the existing channel detector, appends the result to the history and
refreshes the outputs:

    results/figures/live_heatmap.png        rewritten after every sweep
    results/figures/live_spectrum.png       rewritten after every sweep
    data/processed/occupancy_history.csv    rewritten after every sweep
    data/processed/spectrum_history.npz     saved when the session ends
                                            (completed, Ctrl+C or error)

With an interactive matplotlib backend (e.g. WSLg + python3-tk) the figures
are also updated on screen. Without one the PNGs are the live output.

Two candidate outputs per sweep: the instantaneous center candidates of the
current sweep, and stable candidates from center scores averaged over the
last TEMPORAL_WINDOW_SWEEPS sweeps (see src/channel_detector.py, layer 3).

Heatmap orientation: frequency increases left to right, the oldest sweep is
at the top and the newest at the bottom; the y axis shows seconds since the
first sweep started.

Limitation: each row is a sequential LO sweep (~1 s) that is stitched
together, not a simultaneous observation of the whole 100 MHz band.
Receive-only energy sensing; nothing is transmitted or decoded.
"""

from __future__ import annotations

import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Also allow "python scripts/live_monitor.py" from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.channel_detector import (
    ChannelOccupancyReport,
    TemporalCandidateResult,
    TemporalCandidateTracker,
    detect_channel_occupancy,
)
from src.heatmap import (
    OccupancyRecord,
    SpectrumHistory,
    occupancy_matrices,
    reduce_columns_for_display,
    save_occupancy_history_csv,
)
from src.pluto_receiver import PlutoConnectionError, PlutoReceiver
from src.sweep import SpectrumSweeper, StitchedSpectrum, SweepError, stitch_segments
from src.utils import display_path, is_interactive_backend, save_figure_atomic

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
HEATMAP_PATH = FIGURES_DIR / "live_heatmap.png"
SPECTRUM_PATH = FIGURES_DIR / "live_spectrum.png"
SPECTRUM_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "spectrum_history.npz"
OCCUPANCY_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "occupancy_history.csv"

HZ_PER_MHZ = 1e6
BAND_START_MHZ = config.BAND_START_HZ / HZ_PER_MHZ
BAND_STOP_MHZ = config.BAND_STOP_HZ / HZ_PER_MHZ
FIGURE_DPI = 120
STABLE_COLOR = "tab:purple"
INSTANT_COLOR = "tab:orange"


def stable_label(temporal: TemporalCandidateResult) -> str:
    """"last 5" once the window is full, "3/5 sweeps, warming up" before."""
    if temporal.warming_up:
        return f"{temporal.sweeps_used}/{temporal.window_sweeps} sweeps, warming up"
    return f"last {temporal.window_sweeps}"


@contextmanager
def interrupts_ignored() -> Iterator[None]:
    """Ignore Ctrl+C while final results are written."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


# -- live figures ----------------------------------------------------------------


class LiveFigures:
    """Heatmap and latest-spectrum figures, created once and updated in place."""

    def __init__(self, history_length: int) -> None:
        self.history_length = history_length
        self.display_factor = config.HEATMAP_DISPLAY_BIN_HZ // config.STITCH_RESOLUTION_HZ
        if self.display_factor < 1 or (
            config.HEATMAP_DISPLAY_BIN_HZ % config.STITCH_RESOLUTION_HZ
        ):
            raise ValueError(
                "HEATMAP_DISPLAY_BIN_HZ must be a positive multiple of STITCH_RESOLUTION_HZ."
            )
        self.interactive = is_interactive_backend()
        if self.interactive:
            plt.ion()

        self._elapsed = np.empty(0)
        self._build_heatmap()
        self._build_spectrum()

        if self.interactive:
            for figure in (self.heatmap_figure, self.spectrum_figure):
                figure.show()

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _mark_channels(axes: plt.Axes, color: str, alpha: float) -> None:
        for center_hz in config.WIFI_CHANNELS_HZ.values():
            axes.axvline(center_hz / HZ_PER_MHZ, color=color, linestyle=":",
                         linewidth=0.7, alpha=alpha)
        top = axes.secondary_xaxis("top")
        top.set_xticks([hz / HZ_PER_MHZ for hz in config.WIFI_CHANNELS_HZ.values()])
        top.set_xticklabels([str(ch) for ch in config.WIFI_CHANNELS_HZ])
        top.tick_params(labelsize=8, length=2)
        top.set_xlabel("Wi-Fi channel", fontsize=9)

    def _build_heatmap(self) -> None:
        self.heatmap_figure, axes = plt.subplots(figsize=(12, 6.5), layout="constrained")
        cmap = plt.get_cmap("viridis").with_extremes(bad="lightgrey")  # NaN bins
        self.image = axes.imshow(
            np.full((1, 2), np.nan),
            aspect="auto",
            origin="upper",  # oldest sweep at the top
            cmap=cmap,
            vmin=config.HEATMAP_DB_MIN,
            vmax=config.HEATMAP_DB_MAX,
            extent=(BAND_START_MHZ, BAND_STOP_MHZ, 0.5, -0.5),
            interpolation="antialiased",
        )
        colorbar = self.heatmap_figure.colorbar(self.image, ax=axes, extend="both")
        colorbar.set_label("Relative Power (dB)")
        self._mark_channels(axes, color="white", alpha=0.35)

        axes.set_title("2.4 GHz Wi-Fi Activity Over Time")
        axes.set_xlabel("Frequency (MHz)")
        axes.set_ylabel("Time")
        axes.set_xlim(BAND_START_MHZ, BAND_STOP_MHZ)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axes.yaxis.set_major_formatter(FuncFormatter(self._format_row_time))
        self.heatmap_axes = axes

    def _format_row_time(self, value: float, _position: int) -> str:
        row = int(round(value))
        if 0 <= row < self._elapsed.size and abs(value - row) < 1e-6:
            return f"{self._elapsed[row]:.0f} s"
        return ""

    def _build_spectrum(self) -> None:
        self.spectrum_figure, axes = plt.subplots(figsize=(12, 5), layout="constrained")
        self._mark_channels(axes, color="grey", alpha=0.5)
        (self.spectrum_line,) = axes.plot([], [], linewidth=0.6, color="tab:blue",
                                          label="Stitched average spectrum")
        self.floor_line = axes.axhline(np.nan, color="black", linestyle="--", linewidth=1.0,
                                       label="Noise floor")
        self.spectrum_info = axes.text(0.01, 0.97, "", transform=axes.transAxes,
                                       ha="left", va="top", fontsize=9)
        self.candidate_artists: list = []
        self.marker_handles = [
            Line2D([], [], color=STABLE_COLOR, linewidth=2.2, label="Stable candidate"),
            Line2D([], [], color=INSTANT_COLOR, linewidth=1.3, linestyle="--",
                   label="Instant candidate (this sweep)"),
        ]
        axes.set_title("Live 2.4 GHz Spectrum")
        axes.set_xlabel("Frequency (MHz)")
        axes.set_ylabel("Relative Power (dB)")
        axes.set_xlim(BAND_START_MHZ, BAND_STOP_MHZ)
        axes.grid(True, alpha=0.3)
        self.spectrum_axes = axes

    # -- updates --------------------------------------------------------------

    def update(
        self,
        history: SpectrumHistory,
        stitched: StitchedSpectrum,
        report: ChannelOccupancyReport,
        temporal: TemporalCandidateResult,
        sweep_index: int,
        total_sweeps: int,
    ) -> None:
        self._update_heatmap(history)
        self._update_spectrum(stitched, report, temporal, history, sweep_index, total_sweeps)
        if self.interactive:
            self._refresh_screen()

    def _update_heatmap(self, history: SpectrumHistory) -> None:
        _, matrix = reduce_columns_for_display(
            history.frequencies_hz,
            history.power_matrix_db(self.history_length),
            self.display_factor,
        )
        self._elapsed = history.elapsed_seconds(self.history_length)
        rows = matrix.shape[0]
        self.image.set_data(np.ma.masked_invalid(matrix))
        self.image.set_extent((BAND_START_MHZ, BAND_STOP_MHZ, rows - 0.5, -0.5))
        self.heatmap_axes.set_ylim(rows - 0.5, -0.5)
        first, last = history.sweep_indices(self.history_length)[[0, -1]]
        self.heatmap_axes.set_title(
            f"2.4 GHz Wi-Fi Activity Over Time (sweeps {first}–{last}, oldest at top)"
        )

    def _update_spectrum(
        self,
        stitched: StitchedSpectrum,
        report: ChannelOccupancyReport,
        temporal: TemporalCandidateResult,
        history: SpectrumHistory,
        sweep_index: int,
        total_sweeps: int,
    ) -> None:
        """Latest spectrum with stable (solid purple) and instant (dashed orange) markers.

        A channel that is both gets a single solid marker labelled
        "Stable + instant", so identical results are not drawn twice.
        """
        axes = self.spectrum_axes
        power_db = stitched.average_power_db
        self.spectrum_line.set_data(stitched.frequencies_hz / HZ_PER_MHZ, power_db)
        self.floor_line.set_ydata([report.noise_floor_db] * 2)
        self.floor_line.set_label(f"Noise floor {report.noise_floor_db:.1f} dB")

        for artist in self.candidate_artists:
            artist.remove()
        self.candidate_artists = []
        top = axes.get_xaxis_transform()
        stable = set(temporal.candidates)
        instant = set(report.center_candidates)

        for channel in sorted(stable | instant):
            center_mhz = config.WIFI_CHANNELS_HZ[channel] / HZ_PER_MHZ
            if channel in stable:
                label = "Stable + instant" if channel in instant else "Stable"
                self.candidate_artists += [
                    axes.axvline(center_mhz, color=STABLE_COLOR, linewidth=2.2,
                                 alpha=0.7, zorder=1),
                    axes.text(center_mhz, 0.97, f"{label} CH{channel}", transform=top,
                              ha="center", va="top", fontsize=9, color=STABLE_COLOR,
                              fontweight="bold"),
                ]
            else:
                self.candidate_artists += [
                    axes.axvline(center_mhz, color=INSTANT_COLOR, linewidth=1.3,
                                 linestyle="--", alpha=0.9, zorder=1),
                    axes.text(center_mhz, 0.91, f"Instant CH{channel}", transform=top,
                              ha="center", va="top", fontsize=9, color=INSTANT_COLOR),
                ]

        timestamp = history.timestamps()[-1]
        self.spectrum_info.set_text(
            f"Sweep {sweep_index}/{total_sweeps}   {timestamp:%Y-%m-%d %H:%M:%S} UTC\n"
            f"Instant candidates: {report.center_candidates or 'none'}\n"
            f"Stable candidates ({stable_label(temporal)}): {temporal.candidates or 'none'}"
        )
        finite = power_db[np.isfinite(power_db)]
        if finite.size:
            axes.set_ylim(min(finite.min(), report.noise_floor_db) - 3, finite.max() + 8)
        handles = [self.spectrum_line, self.floor_line, *self.marker_handles]
        axes.legend(handles=handles, loc="upper right", fontsize=9)

    def _refresh_screen(self) -> None:
        try:
            for figure in (self.heatmap_figure, self.spectrum_figure):
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:  # display problems must not stop the measurement
            print(f"  Live display disabled ({type(exc).__name__}: {exc}); PNGs still updated.")
            self.interactive = False

    def save(self) -> None:
        """Overwrite live_heatmap.png and live_spectrum.png (atomic rename)."""
        save_figure_atomic(self.heatmap_figure, HEATMAP_PATH, dpi=FIGURE_DPI)
        save_figure_atomic(self.spectrum_figure, SPECTRUM_PATH, dpi=FIGURE_DPI)


# -- session ---------------------------------------------------------------------


def run_sweep(sweeper: SpectrumSweeper) -> tuple[StitchedSpectrum, datetime, float, float]:
    """One full-band sweep using the existing sweep and stitching code.

    Returns:
        ``(stitched, start timestamp UTC, start time.monotonic(), duration s)``.
    """
    timestamp = datetime.now(timezone.utc)
    start = time.monotonic()
    stitched = stitch_segments(sweeper.run())
    return stitched, timestamp, start, time.monotonic() - start


def history_metadata(
    receiver: PlutoReceiver,
    sweeper: SpectrumSweeper,
    gain: tuple[str, float],
    records: list[OccupancyRecord],
    report: Optional[ChannelOccupancyReport],
    stop_reason: str,
) -> dict:
    """Receiver, sweep, detector and temporal settings plus per-sweep channel matrices."""
    metadata = {
        "sample_rate": receiver.sample_rate,
        "rf_bandwidth": receiver.rf_bandwidth,
        "fft_size": receiver.buffer_size,
        "rx_gain_control_mode": gain[0],
        "rx_gain_db": gain[1],
        "sweep_centers_hz": np.array(sweeper.center_frequencies_hz, dtype=np.int64),
        "captures_per_center": sweeper.num_captures,
        "stitch_resolution_hz": config.STITCH_RESOLUTION_HZ,
        "band_start_hz": config.BAND_START_HZ,
        "band_stop_hz": config.BAND_STOP_HZ,
        "monitor_interval_seconds": config.MONITOR_INTERVAL_SECONDS,
        "temporal_window_sweeps": config.TEMPORAL_WINDOW_SWEEPS,
        "temporal_min_support": config.TEMPORAL_MIN_SUPPORT,
        "stop_reason": stop_reason,
        **occupancy_matrices(records),
    }
    if report is not None:
        metadata.update({f"detector_{k}": v for k, v in report.settings.__dict__.items()})
    return metadata


def print_candidate_statistics(records: list[OccupancyRecord]) -> None:
    """Occurrence counts of instant candidates and the final stable result."""
    counts = Counter(c for r in records for c in r.instant_center_candidates)
    empty = sum(1 for r in records if not r.instant_center_candidates)
    print()
    print("Instant candidate occurrence counts:")
    for channel in sorted(counts):
        print(f"CH{channel}: {counts[channel]}")
    print(f"None: {empty}")

    stable_counts = Counter(c for r in records for c in r.stable_center_candidates)
    print()
    print("Stable candidate occurrence counts:")
    for channel in sorted(stable_counts):
        print(f"CH{channel}: {stable_counts[channel]}")
    print(f"None: {sum(1 for r in records if not r.stable_center_candidates)}")

    last = records[-1]
    window = (f"{last.stable_sweeps_used}/{last.stable_window_sweeps} sweeps, warming up"
              if last.stable_warming_up else f"last {last.stable_window_sweeps} sweeps")
    print()
    print(f"Final stable candidate ({window}):")
    print(list(last.stable_center_candidates))


def print_startup_information(
    receiver: PlutoReceiver,
    sweeper: SpectrumSweeper,
    gain: tuple[str, float],
    num_sweeps: int,
) -> None:
    """Concise session header for demonstrations."""
    print("ADALM-PLUTO Part 1 Wi-Fi Spectrum Monitor")
    print()
    print(f"Device URI: {receiver.uri}")
    print(f"Band: {BAND_START_MHZ:.0f}–{BAND_STOP_MHZ:.0f} MHz")
    print(f"Sample Rate: {receiver.sample_rate / HZ_PER_MHZ:g} MSPS")
    print(f"RX Gain: {gain[1]:g} dB {gain[0]}")
    print(f"FFT Size: {receiver.buffer_size}")
    print(f"Captures per LO center: {sweeper.num_captures}")
    print(f"Sweep centers: {len(sweeper.center_frequencies_hz)}")
    print(f"Temporal window: {config.TEMPORAL_WINDOW_SWEEPS} sweeps")
    print()
    print(f"Sweeps: {num_sweeps} (about {num_sweeps * config.EXPECTED_SWEEP_SECONDS:.0f} s "
          "of sweeping plus plot updates)")
    print("Receive-only energy sensing: nothing is transmitted, no packets are decoded.")
    print("Each heatmap row is a sequential ~1 s LO sweep, not a simultaneous 100 MHz view.")
    print(f"Live figures: {display_path(SPECTRUM_PATH)}, {display_path(HEATMAP_PATH)}")
    print("Press Ctrl+C to stop early; completed sweeps are saved.")


def main() -> int:
    """Run a live monitoring session with the settings in src/config.py.

    Returns:
        Process exit code: 0 if all sweeps completed, 1 if stopped early or failed.
    """
    num_sweeps = int(config.MONITOR_NUM_SWEEPS)
    interval_s = float(config.MONITOR_INTERVAL_SECONDS)
    history_length = int(config.HEATMAP_HISTORY_LENGTH)
    if num_sweeps < 1 or history_length < 1 or interval_s < 0:
        print("Error: MONITOR_NUM_SWEEPS and HEATMAP_HISTORY_LENGTH must be >= 1, "
              "MONITOR_INTERVAL_SECONDS >= 0.")
        return 1

    centers = config.SWEEP_CENTER_FREQUENCIES_HZ
    receiver = PlutoReceiver(uri=config.PLUTO_URI, center_freq=centers[0])
    sweeper = SpectrumSweeper(receiver, centers)  # default log: no per-capture output

    try:
        receiver.connect()
        receiver.configure()
        gain = sweeper.check_fixed_gain()
    except (PlutoConnectionError, SweepError) as exc:
        receiver.close()
        print(f"Error: {exc}")
        print("Run `python -m scripts.test_pluto` first to debug the connection.")
        return 1

    print_startup_information(receiver, sweeper, gain, num_sweeps)

    try:
        tracker = TemporalCandidateTracker()
        figures = LiveFigures(history_length)
    except ValueError as exc:
        receiver.close()
        print(f"Error: {exc}")
        return 1
    if not figures.interactive:
        print(f"No plot window available ('{matplotlib.get_backend()}' backend): "
              "open the PNG files to follow the live output.")

    history: Optional[SpectrumHistory] = None
    records: list[OccupancyRecord] = []
    last_report: Optional[ChannelOccupancyReport] = None
    update_times: list[float] = []
    stop_reason = "completed"
    session_start = time.monotonic()
    sweep_number = 0

    try:
        for sweep_number in range(1, num_sweeps + 1):
            iteration_start = time.monotonic()
            try:
                stitched, timestamp, start_mono, duration_s = run_sweep(sweeper)
                report = detect_channel_occupancy(
                    stitched.frequencies_hz,
                    stitched.average_power_linear,
                    stitched.average_power_db,
                )
                temporal = tracker.update(report)
            except (SweepError, PlutoConnectionError, ValueError) as exc:
                print(f"\nSweep {sweep_number}/{num_sweeps} failed: {exc}")
                stop_reason = "error"
                break

            update_start = time.monotonic()
            elapsed_s = start_mono - session_start
            record = OccupancyRecord.from_report(
                sweep_number, timestamp, elapsed_s, duration_s, stitched.coverage_percent,
                report, temporal)
            if history is None:
                history = SpectrumHistory(stitched.frequencies_hz)
            history.append(stitched.frequencies_hz, stitched.average_power_db, timestamp,
                           report.noise_floor_db, duration_s, elapsed_s)
            records.append(record)
            last_report = report

            figures.update(history, stitched, report, temporal, sweep_number, num_sweeps)
            figures.save()
            save_occupancy_history_csv(OCCUPANCY_HISTORY_PATH, records)
            update_times.append(time.monotonic() - update_start)

            print(f"\nSweep {sweep_number}/{num_sweeps}   ({duration_s:.2f} s, "
                  "spectrum + heatmap updated)")
            if stitched.uncovered_bins:
                print(f"  WARNING: {stitched.uncovered_bins} uncovered bins (kept as NaN)")
            print(f"  Noise floor:       {report.noise_floor_db:.1f} dB")
            print(f"  Energy windows:    {report.energy_occupied_channels}")
            print(f"  Instant candidate: {report.center_candidates}")
            print(f"  Stable candidate:  {temporal.candidates}   ({stable_label(temporal)})")

            remaining = interval_s - (time.monotonic() - iteration_start)
            if interval_s > 0 and remaining > 0 and sweep_number < num_sweeps:
                time.sleep(remaining)
    except KeyboardInterrupt:
        stop_reason = "user"
        print("\n\nMonitoring stopped by user.")
    finally:
        receiver.close()
        total_time_s = time.monotonic() - session_start
        completed = len(records)
        with interrupts_ignored():
            if history is not None and completed:
                print("Saving results (Ctrl+C is ignored until saving is finished)...")
                history.save_npz(
                    SPECTRUM_HISTORY_PATH,
                    history_metadata(receiver, sweeper, gain, records, last_report, stop_reason),
                )
                save_occupancy_history_csv(OCCUPANCY_HISTORY_PATH, records)
                figures.save()

    if stop_reason == "user":
        print(f"Saved {completed} completed sweeps.")
    elif stop_reason == "error":
        print(f"Monitoring stopped after a failure in sweep {sweep_number}. "
              f"Saved {completed} completed sweeps.")
    else:
        print("\nMonitoring completed.")

    if not completed:
        print("No sweep was completed; no history files were written.")
        plt.close("all")
        return 1

    durations = history.sweep_durations_s()
    print()
    print(f"Sweeps completed: {completed}")
    print(f"Total monitoring time: {total_time_s:.1f} s")
    print(f"Average sweep duration: {durations.mean():.2f} s")
    print(f"Min sweep duration: {durations.min():.2f} s")
    print(f"Max sweep duration: {durations.max():.2f} s")
    print(f"Average update time (history, plots, CSV): {np.mean(update_times):.2f} s")
    print_candidate_statistics(records)
    print()
    print("Spectrum history saved to:")
    print(display_path(SPECTRUM_HISTORY_PATH))
    print()
    print("Occupancy history saved to:")
    print(display_path(OCCUPANCY_HISTORY_PATH))
    print()
    print("Heatmap saved to:")
    print(display_path(HEATMAP_PATH))
    print()
    print("Latest spectrum saved to:")
    print(display_path(SPECTRUM_PATH))

    if figures.interactive:
        print("\nClose the plot windows to exit.")
        plt.ioff()
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    plt.close("all")
    return 0 if stop_reason == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
