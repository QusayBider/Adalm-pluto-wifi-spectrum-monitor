"""Time history of stitched full-band spectra and channel occupancy.

No hardware access and no plotting: this module only stores, validates and
saves the results of repeated sweeps.

Row layout: row 0 is the oldest stored sweep, the last row the newest.
Each row is the stitched average spectrum (relative dB) of one full-band
sweep; uncovered bins stay NaN.

IMPORTANT LIMITATION - SEQUENTIAL SWEEPS:
    The ADALM-PLUTO does not observe the 100 MHz band simultaneously. Each
    row is built from consecutive LO steps (about 1 s per sweep) that are
    stitched together. Wi-Fi traffic is bursty, so activity can change while
    a single row is being measured; different parts of one row were measured
    at slightly different times. The row timestamp is the sweep start time.
"""

from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .channel_detector import ChannelOccupancyReport, TemporalCandidateResult

# Frequency grids from make_band_grid() are deterministic; this only absorbs
# float noise while still rejecting any real grid change.
GRID_TOLERANCE_HZ = 1e-3


def _require_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware (use datetime.now(timezone.utc)).")
    return timestamp.astimezone(timezone.utc)


class SpectrumHistory:
    """Rows of stitched spectra on one fixed frequency grid.

    Args:
        frequencies_hz: The frequency grid every appended spectrum must use.
        max_length: Keep only the newest ``max_length`` rows (rolling
            history). ``None`` keeps every row.
    """

    def __init__(self, frequencies_hz: np.ndarray, max_length: Optional[int] = None) -> None:
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size < 2 or np.any(np.diff(frequencies) <= 0):
            raise ValueError("frequencies_hz must be a strictly ascending 1-D array.")
        if max_length is not None and max_length < 1:
            raise ValueError(f"max_length must be >= 1 or None, got {max_length}.")

        self._frequencies = frequencies
        self.max_length = max_length
        self._power_db: deque[np.ndarray] = deque(maxlen=max_length)
        self._timestamps: deque[datetime] = deque(maxlen=max_length)
        self._noise_floor_db: deque[float] = deque(maxlen=max_length)
        self._durations_s: deque[float] = deque(maxlen=max_length)
        self._sweep_index: deque[int] = deque(maxlen=max_length)
        self._elapsed_s: deque[float] = deque(maxlen=max_length)
        self._start_time: Optional[datetime] = None
        self._appended = 0

    # -- adding rows ----------------------------------------------------------

    def append(
        self,
        frequencies_hz: np.ndarray,
        power_db: np.ndarray,
        timestamp: datetime,
        noise_floor_db: float = float("nan"),
        sweep_duration_s: float = float("nan"),
        elapsed_seconds: Optional[float] = None,
    ) -> int:
        """Add one sweep. Returns its 1-based sweep index within the session.

        Args:
            elapsed_seconds: Seconds since the session start, preferably from
                ``time.monotonic()``. The wall clock can step (NTP, WSL time
                sync), so it is only used when this is not given.

        Raises:
            ValueError: if the frequency grid differs from the history's grid,
                the power array has the wrong shape or contains +/-Inf, or the
                timestamp is naive or older than the previous one.
        """
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.shape != self._frequencies.shape or not np.allclose(
            frequencies, self._frequencies, rtol=0.0, atol=GRID_TOLERANCE_HZ
        ):
            raise ValueError(
                "Frequency grid of the new sweep differs from the history grid "
                f"({frequencies.size} vs {self._frequencies.size} bins)."
            )
        power = np.asarray(power_db, dtype=np.float64)
        if power.shape != self._frequencies.shape:
            raise ValueError(
                f"power_db has shape {power.shape}, expected {self._frequencies.shape}."
            )
        if np.any(np.isinf(power)):
            raise ValueError("power_db contains +/-Inf; uncovered bins must be NaN.")

        timestamp = _require_utc(timestamp)
        if self._start_time is None:
            self._start_time = timestamp
        if elapsed_seconds is None:
            elapsed_seconds = (timestamp - self._start_time).total_seconds()
        if self._elapsed_s and elapsed_seconds < self._elapsed_s[-1]:
            raise ValueError("Elapsed time must not go backwards.")

        self._appended += 1
        self._elapsed_s.append(float(elapsed_seconds))
        self._power_db.append(power.copy())
        self._timestamps.append(timestamp)
        self._noise_floor_db.append(float(noise_floor_db))
        self._durations_s.append(float(sweep_duration_s))
        self._sweep_index.append(self._appended)
        return self._appended

    # -- reading --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._power_db)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self._frequencies.copy()

    @property
    def start_time(self) -> Optional[datetime]:
        """Timestamp of the first sweep ever appended (survives rolling)."""
        return self._start_time

    @property
    def total_sweeps(self) -> int:
        """Number of sweeps appended, including rows dropped by rolling."""
        return self._appended

    def _last(self, values: Sequence[Any], last_n: Optional[int]) -> list[Any]:
        values = list(values)
        return values if last_n is None else values[-last_n:]

    def power_matrix_db(self, last_n: Optional[int] = None) -> np.ndarray:
        """``(rows, bins)`` matrix, oldest row first. Empty ``(0, bins)`` if none."""
        rows = self._last(self._power_db, last_n)
        if not rows:
            return np.empty((0, self._frequencies.size))
        return np.vstack(rows)

    def timestamps(self, last_n: Optional[int] = None) -> list[datetime]:
        return self._last(self._timestamps, last_n)

    def elapsed_seconds(self, last_n: Optional[int] = None) -> np.ndarray:
        """Seconds from the session start to each row's sweep start."""
        return np.array(self._last(self._elapsed_s, last_n), dtype=np.float64)

    def noise_floor_db(self, last_n: Optional[int] = None) -> np.ndarray:
        return np.array(self._last(self._noise_floor_db, last_n), dtype=np.float64)

    def sweep_durations_s(self, last_n: Optional[int] = None) -> np.ndarray:
        return np.array(self._last(self._durations_s, last_n), dtype=np.float64)

    def sweep_indices(self, last_n: Optional[int] = None) -> np.ndarray:
        return np.array(self._last(self._sweep_index, last_n), dtype=np.int64)

    # -- persistence ----------------------------------------------------------

    def save_npz(self, path: Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Save every stored row plus optional metadata arrays/scalars.

        Written to a temporary file first and then renamed, so an interrupted
        save never leaves a truncated file behind.
        """
        timestamps = self.timestamps()
        arrays: dict[str, Any] = {
            "frequencies_hz": self._frequencies,
            "power_history_db": self.power_matrix_db().astype(np.float32),
            "timestamps_utc": np.array([t.isoformat() for t in timestamps], dtype=str),
            "timestamps_unix_s": np.array([t.timestamp() for t in timestamps]),
            "elapsed_seconds": self.elapsed_seconds(),
            "noise_floor_history_db": self.noise_floor_db(),
            "sweep_duration_seconds": self.sweep_durations_s(),
            "sweep_index": self.sweep_indices(),
        }
        for key, value in (metadata or {}).items():
            if key in arrays:
                raise ValueError(f"Metadata key '{key}' collides with a history array.")
            arrays[key] = value

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load_npz(cls, path: Path) -> "SpectrumHistory":
        """Rebuild a history (unlimited length) from :meth:`save_npz` output."""
        with np.load(path) as data:
            history = cls(data["frequencies_hz"])
            for row, stamp, floor, duration, elapsed in zip(
                data["power_history_db"],
                data["timestamps_utc"],
                data["noise_floor_history_db"],
                data["sweep_duration_seconds"],
                data["elapsed_seconds"],
            ):
                history.append(
                    data["frequencies_hz"], row, datetime.fromisoformat(str(stamp)),
                    floor, duration, elapsed,
                )
        return history


def reduce_columns_for_display(
    frequencies_hz: np.ndarray,
    power_matrix_db: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average every ``factor`` adjacent bins in linear power (display only).

    A 20000-bin row is wider than any screen or PNG, and letting matplotlib
    resample it is slow and averages in dB. Averaging linear power over a few
    5 kHz cells gives a physically meaningful, faster image. NaN bins are
    ignored; a column with only NaN bins stays NaN. Trailing bins that do not
    fill a whole group are dropped. The stored history is not modified.

    Returns:
        ``(column_center_frequencies_hz, reduced_matrix_db)``.
    """
    matrix = np.atleast_2d(np.asarray(power_matrix_db, dtype=np.float64))
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}.")
    if factor == 1:
        return frequencies, matrix

    columns = matrix.shape[1] // factor
    used = columns * factor
    linear = 10.0 ** (matrix[:, :used].reshape(matrix.shape[0], columns, factor) / 10.0)
    valid = np.isfinite(linear)
    counts = valid.sum(axis=2)
    sums = np.where(valid, linear, 0.0).sum(axis=2)
    reduced = np.full(sums.shape, np.nan)
    np.divide(sums, counts, out=reduced, where=counts > 0)
    reduced_db = 10.0 * np.log10(reduced)
    reduced_frequencies = frequencies[:used].reshape(columns, factor).mean(axis=1)
    return reduced_frequencies, reduced_db


# -- channel occupancy over time ------------------------------------------------


@dataclass(frozen=True)
class OccupancyRecord:
    """Channel-analysis summary of one sweep.

    ``center_candidates`` holds the instantaneous candidates of this sweep
    (``instant_center_candidates`` is an alias). ``stable_center_candidates``
    come from the temporal consensus over the recent sweeps, if available.
    Scores are ordered like ``channel_numbers``.
    """

    sweep_index: int
    timestamp_utc: datetime
    elapsed_seconds: float
    sweep_duration_seconds: float
    noise_floor_db: float
    coverage_percent: float
    channel_numbers: tuple[int, ...]
    energy_occupied_channels: tuple[int, ...]
    center_candidates: tuple[int, ...]
    instant_center_scores: tuple[float, ...] = ()
    stable_center_candidates: tuple[int, ...] = ()
    temporal_center_scores: tuple[float, ...] = ()
    stable_sweeps_used: int = 0
    stable_window_sweeps: int = 0

    @property
    def instant_center_candidates(self) -> tuple[int, ...]:
        return self.center_candidates

    @property
    def stable_warming_up(self) -> bool:
        return self.stable_sweeps_used < self.stable_window_sweeps

    @classmethod
    def from_report(
        cls,
        sweep_index: int,
        timestamp: datetime,
        elapsed_seconds: float,
        sweep_duration_seconds: float,
        coverage_percent: float,
        report: ChannelOccupancyReport,
        temporal: Optional[TemporalCandidateResult] = None,
    ) -> "OccupancyRecord":
        """Summarise one sweep's channel report (and optional stable result)."""
        channel_numbers = tuple(r.channel_number for r in report.results)
        stable_fields: dict[str, Any] = {}
        if temporal is not None:
            temporal_scores = temporal.temporal_scores
            stable_fields = {
                "stable_center_candidates": tuple(temporal.candidates),
                "temporal_center_scores": tuple(temporal_scores[c] for c in channel_numbers),
                "stable_sweeps_used": temporal.sweeps_used,
                "stable_window_sweeps": temporal.window_sweeps,
            }
        return cls(
            sweep_index=sweep_index,
            timestamp_utc=_require_utc(timestamp),
            elapsed_seconds=float(elapsed_seconds),
            sweep_duration_seconds=float(sweep_duration_seconds),
            noise_floor_db=report.noise_floor_db,
            coverage_percent=float(coverage_percent),
            channel_numbers=channel_numbers,
            energy_occupied_channels=tuple(report.energy_occupied_channels),
            center_candidates=tuple(report.center_candidates),
            instant_center_scores=tuple(s.center_score for s in report.center_scores),
            **stable_fields,
        )


def _channel_list(channels: Sequence[int]) -> str:
    return ",".join(str(c) for c in channels)


def occupancy_matrices(records: Sequence[OccupancyRecord]) -> dict[str, np.ndarray]:
    """``(sweeps, channels)`` matrices of the per-sweep channel results.

    Boolean: energy occupancy, instant candidates, stable candidates.
    Float: instant and temporal center scores (NaN where not available).
    """
    channel_numbers = records[0].channel_numbers if records else ()
    shape = (len(records), len(channel_numbers))

    def flags(attribute: str) -> np.ndarray:
        return np.array(
            [[c in getattr(r, attribute) for c in channel_numbers] for r in records],
            dtype=bool,
        ).reshape(shape)

    def scores(attribute: str) -> np.ndarray:
        rows = [getattr(r, attribute) or (float("nan"),) * len(channel_numbers) for r in records]
        return np.array(rows, dtype=np.float64).reshape(shape)

    return {
        "channel_numbers": np.array(channel_numbers, dtype=np.int64),
        "energy_occupied_history": flags("energy_occupied_channels"),
        "center_candidate_history": flags("center_candidates"),
        "stable_candidate_history": flags("stable_center_candidates"),
        "instant_center_score_history": scores("instant_center_scores"),
        "temporal_center_score_history": scores("temporal_center_scores"),
    }


def save_occupancy_history_csv(path: Path, records: Sequence[OccupancyRecord]) -> None:
    """One row per sweep. Channel lists are comma-separated, e.g. "4,5,6".

    ``center_candidates`` and ``instant_center_candidates`` both hold the
    instantaneous result (the first is kept for compatibility);
    ``stable_center_candidates`` holds the temporal consensus. One boolean
    column per channel is written for energy occupancy (``chN_energy``),
    instantaneous candidates (``chN_candidate``) and stable candidates
    (``chN_stable_candidate``). Center scores per sweep are saved in
    spectrum_history.npz. Written via a temporary file and rename.
    """
    channel_numbers = records[0].channel_numbers if records else ()
    columns = [
        "timestamp_utc",
        "sweep_index",
        "elapsed_seconds",
        "sweep_duration_seconds",
        "noise_floor_db",
        "coverage_percent",
        "energy_occupied_channels",
        "center_candidates",
        "instant_center_candidates",
        "stable_center_candidates",
        "stable_sweeps_used",
        "stable_warming_up",
        *[f"ch{c}_energy" for c in channel_numbers],
        *[f"ch{c}_candidate" for c in channel_numbers],
        *[f"ch{c}_stable_candidate" for c in channel_numbers],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in records:
            row = {
                "timestamp_utc": r.timestamp_utc.isoformat(),
                "sweep_index": r.sweep_index,
                "elapsed_seconds": f"{r.elapsed_seconds:.3f}",
                "sweep_duration_seconds": f"{r.sweep_duration_seconds:.3f}",
                "noise_floor_db": f"{r.noise_floor_db:.3f}",
                "coverage_percent": f"{r.coverage_percent:.2f}",
                "energy_occupied_channels": _channel_list(r.energy_occupied_channels),
                "center_candidates": _channel_list(r.center_candidates),
                "instant_center_candidates": _channel_list(r.instant_center_candidates),
                "stable_center_candidates": _channel_list(r.stable_center_candidates),
                "stable_sweeps_used": r.stable_sweeps_used,
                "stable_warming_up": r.stable_warming_up,
            }
            for c in channel_numbers:
                row[f"ch{c}_energy"] = c in r.energy_occupied_channels
                row[f"ch{c}_candidate"] = c in r.center_candidates
                row[f"ch{c}_stable_candidate"] = c in r.stable_center_candidates
            writer.writerow(row)
    os.replace(temporary, path)
