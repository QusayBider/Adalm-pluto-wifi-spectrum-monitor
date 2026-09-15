"""Energy-based analysis of the 2.4 GHz Wi-Fi channels.

Input is the stitched full-band spectrum (frequencies, average power). This
module never touches the SDR and never plots.

IMPORTANT LIMITATION - ENERGY DETECTION ONLY:
    Everything here measures RF energy at frequencies. No 802.11 frame is
    decoded, so nothing here can prove that the energy is Wi-Fi, nor identify
    an access point. Bluetooth, microwave ovens, video senders, other ISM
    devices or receiver spurs are measured in exactly the same way.

Two layers of interpretation:

1. ENERGY OCCUPANCY (per channel window)
   "There is significant RF energy in this standard channel frequency
   window." Channels are 5 MHz apart but ~20 MHz wide, so one 20 MHz signal
   makes several neighbouring windows energy-occupied.

       noise_floor_db  = percentile(covered bins in dB, NOISE_FLOOR_PERCENTILE)
       power_excess_db = median window power (dB) - noise_floor_db
       active fraction = share of window bins >= noise floor + ACTIVE_BIN_MARGIN_DB
       ENERGY OCCUPIED if power_excess_db >= CHANNEL_POWER_MARGIN_DB
                       or active fraction >= MIN_ACTIVE_BIN_FRACTION

2. CENTER CANDIDATES (across channels)
   "The measured broadband energy is most consistent with a signal centered
   near this standard channel frequency."

       core            = center +/- CORE_HALF_BANDWIDTH_HZ
       center_score    = max(core median - noise floor, 0) * core active fraction
       local maximum   = score >= all channels within CANDIDATE_NEIGHBOR_CHANNELS
       candidate       = local maximum and score >= MIN_CENTER_SCORE,
                         after non-maximum suppression with
                         MIN_CANDIDATE_SEPARATION_HZ

   Two real networks whose centers are closer than the separation (or whose
   spectra merge into one plateau) cannot be told apart by energy alone.

3. STABLE (TEMPORAL) CENTER CANDIDATES (across recent sweeps)
   The instantaneous candidate of a single sweep fluctuates, typically
   between neighbouring channels, because Wi-Fi traffic is bursty, adjacent
   channels overlap, and the band is measured by sequential LO tuning.

       temporal_score[ch] = mean(center_score[ch] over the last TEMPORAL_WINDOW_SWEEPS)
       support[ch]        = sweeps in that window with center_score[ch] >= MIN_CENTER_SCORE
       stable candidate   = layer-2 rules applied to temporal_score, and
                            support >= min(TEMPORAL_MIN_SUPPORT, sweeps available)

   Averaging reduces short-term fluctuations of the interpretation. It is
   still energy sensing: it does not decode packets or identify APs.
   Energy occupancy (layer 1) is never temporally filtered.

Peak power is reported as a diagnostic only; it never drives a decision.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from . import config
from .spectrum import power_to_db

HZ_PER_MHZ = 1e6

STATUS_OCCUPIED = "OCCUPIED"
STATUS_FREE = "FREE"
STATUS_NO_DATA = "NO DATA"


@dataclass(frozen=True)
class WiFiChannel:
    """One standard channel and its analysis window."""

    number: int
    center_frequency_hz: int
    bandwidth_hz: int

    @property
    def start_hz(self) -> float:
        return self.center_frequency_hz - self.bandwidth_hz / 2

    @property
    def stop_hz(self) -> float:
        return self.center_frequency_hz + self.bandwidth_hz / 2


@dataclass(frozen=True)
class DetectionSettings:
    """Tunable detector parameters. Defaults come from config.py."""

    noise_floor_percentile: float = config.NOISE_FLOOR_PERCENTILE
    active_bin_margin_db: float = config.ACTIVE_BIN_MARGIN_DB
    channel_power_margin_db: float = config.CHANNEL_POWER_MARGIN_DB
    min_active_bin_fraction: float = config.MIN_ACTIVE_BIN_FRACTION
    core_half_bandwidth_hz: float = config.CORE_HALF_BANDWIDTH_HZ
    min_center_score: float = config.MIN_CENTER_SCORE
    candidate_neighbor_channels: int = config.CANDIDATE_NEIGHBOR_CHANNELS
    min_candidate_separation_hz: float = config.MIN_CANDIDATE_SEPARATION_HZ

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise_floor_percentile <= 100.0:
            raise ValueError("noise_floor_percentile must be within 0..100.")
        if not 0.0 < self.min_active_bin_fraction <= 1.0:
            raise ValueError("min_active_bin_fraction must be within (0, 1].")
        if self.core_half_bandwidth_hz <= 0:
            raise ValueError("core_half_bandwidth_hz must be positive.")
        if self.candidate_neighbor_channels < 1:
            raise ValueError("candidate_neighbor_channels must be >= 1.")
        if self.min_candidate_separation_hz < 0:
            raise ValueError("min_candidate_separation_hz must not be negative.")


@dataclass(frozen=True)
class WiFiChannelResult:
    """Energy-occupancy metrics for one channel window. Powers in relative dB."""

    channel_number: int
    center_frequency_hz: int
    window_start_hz: float
    window_stop_hz: float
    valid_bins: int
    mean_power_db: float
    median_power_db: float
    peak_power_db: float
    noise_floor_db: float
    power_excess_db: float
    active_bin_fraction: float
    energy_occupied: bool
    status: str
    decision_reason: str

    @property
    def occupied(self) -> bool:
        """Backwards-compatible alias of :attr:`energy_occupied`."""
        return self.energy_occupied


@dataclass(frozen=True)
class ChannelCenterScore:
    """Center-candidate metrics for one channel. Powers in relative dB.

    Attributes:
        core_mean_power_db: Mean of the core in linear power, as dB.
        core_median_excess_db: Core median minus the noise floor.
        core_active_fraction: Share of core bins above the active threshold.
        center_score: ``max(core_median_excess_db, 0) * core_active_fraction``
            (NaN without data).
        is_local_maximum: Score >= all channels in the neighbourhood.
        center_candidate: Final decision after threshold and suppression.
        note: Why a local maximum was not accepted, if applicable.
    """

    channel_number: int
    center_frequency_hz: int
    core_start_hz: float
    core_stop_hz: float
    core_valid_bins: int
    core_mean_power_db: float
    core_median_power_db: float
    core_median_excess_db: float
    core_active_fraction: float
    center_score: float
    is_local_maximum: bool = False
    center_candidate: bool = False
    note: str = ""


@dataclass(frozen=True)
class ChannelOccupancyReport:
    """Both interpretation layers for one full-band spectrum."""

    noise_floor_db: float
    active_threshold_db: float
    settings: DetectionSettings
    results: list[WiFiChannelResult] = field(default_factory=list)
    center_scores: list[ChannelCenterScore] = field(default_factory=list)

    @property
    def energy_occupied_channels(self) -> list[int]:
        return [r.channel_number for r in self.results if r.status == STATUS_OCCUPIED]

    @property
    def occupied_channels(self) -> list[int]:
        """Backwards-compatible alias of :attr:`energy_occupied_channels`."""
        return self.energy_occupied_channels

    @property
    def free_channels(self) -> list[int]:
        return [r.channel_number for r in self.results if r.status == STATUS_FREE]

    @property
    def no_data_channels(self) -> list[int]:
        return [r.channel_number for r in self.results if r.status == STATUS_NO_DATA]

    @property
    def center_candidates(self) -> list[int]:
        return [s.channel_number for s in self.center_scores if s.center_candidate]


def wifi_channels(
    centers_hz: Mapping[int, int] = config.WIFI_CHANNELS_HZ,
    bandwidth_hz: int = config.WIFI_CHANNEL_BANDWIDTH_HZ,
) -> list[WiFiChannel]:
    """Channel definitions from config, ordered by channel number."""
    return [
        WiFiChannel(number, int(center), int(bandwidth_hz))
        for number, center in sorted(centers_hz.items())
    ]


def estimate_noise_floor_db(
    power_db: np.ndarray,
    percentile: float = config.NOISE_FLOOR_PERCENTILE,
) -> float:
    """Robust noise-floor estimate: a low percentile of all covered bins.

    Bins carrying signal sit in the upper part of the distribution, so a low
    percentile ignores them as long as most of the band is noise. NaN
    (uncovered) bins are ignored.

    Raises:
        ValueError: if there are no finite bins.
    """
    finite = np.asarray(power_db, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot estimate the noise floor: no covered spectrum bins.")
    return float(np.percentile(finite, percentile))


def _validate_spectrum(
    frequencies_hz: np.ndarray,
    average_power_linear: np.ndarray,
    average_power_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    linear = np.asarray(average_power_linear, dtype=np.float64)
    power_db = np.asarray(average_power_db, dtype=np.float64)

    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array.")
    if linear.shape != frequencies.shape or power_db.shape != frequencies.shape:
        raise ValueError(
            f"Spectrum arrays differ in shape: frequencies {frequencies.shape}, "
            f"linear {linear.shape}, dB {power_db.shape}."
        )
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies_hz must be strictly ascending.")
    if not np.array_equal(np.isfinite(linear), np.isfinite(power_db)):
        raise ValueError("Linear and dB spectra do not have the same covered bins.")
    return frequencies, linear, power_db


# -- layer 1: energy occupancy ------------------------------------------------


def analyze_channel(
    channel: WiFiChannel,
    frequencies_hz: np.ndarray,
    average_power_linear: np.ndarray,
    average_power_db: np.ndarray,
    noise_floor_db: float,
    settings: DetectionSettings,
) -> WiFiChannelResult:
    """Energy-occupancy metrics and decision for one channel window.

    The window is clipped to the measured frequencies. Uncovered (NaN) bins
    are ignored; if no covered bins remain the status is NO DATA.
    """
    in_window = (frequencies_hz >= channel.start_hz) & (frequencies_hz <= channel.stop_hz)
    covered = in_window & np.isfinite(average_power_db)

    window_start = float(max(channel.start_hz, frequencies_hz[0]))
    window_stop = float(min(channel.stop_hz, frequencies_hz[-1]))
    valid_bins = int(np.count_nonzero(covered))

    if valid_bins == 0:
        nan = float("nan")
        return WiFiChannelResult(
            channel.number, channel.center_frequency_hz, window_start, window_stop,
            0, nan, nan, nan, noise_floor_db, nan, nan,
            energy_occupied=False, status=STATUS_NO_DATA, decision_reason="no covered bins",
        )

    channel_db = average_power_db[covered]
    # Mean power is averaged in linear power; median and peak are unaffected
    # by the choice of domain.
    mean_power_db = float(power_to_db(np.mean(average_power_linear[covered])))
    median_power_db = float(np.median(channel_db))
    peak_power_db = float(np.max(channel_db))

    power_excess_db = median_power_db - noise_floor_db
    active_threshold_db = noise_floor_db + settings.active_bin_margin_db
    active_bin_fraction = float(np.count_nonzero(channel_db >= active_threshold_db) / valid_bins)

    reasons = []
    if power_excess_db >= settings.channel_power_margin_db:
        reasons.append("median excess")
    if active_bin_fraction >= settings.min_active_bin_fraction:
        reasons.append("active bins")
    energy_occupied = bool(reasons)

    return WiFiChannelResult(
        channel_number=channel.number,
        center_frequency_hz=channel.center_frequency_hz,
        window_start_hz=window_start,
        window_stop_hz=window_stop,
        valid_bins=valid_bins,
        mean_power_db=mean_power_db,
        median_power_db=median_power_db,
        peak_power_db=peak_power_db,
        noise_floor_db=noise_floor_db,
        power_excess_db=power_excess_db,
        active_bin_fraction=active_bin_fraction,
        energy_occupied=energy_occupied,
        status=STATUS_OCCUPIED if energy_occupied else STATUS_FREE,
        decision_reason=" + ".join(reasons),
    )


# -- layer 2: center candidates -----------------------------------------------


def score_channel_center(
    channel: WiFiChannel,
    frequencies_hz: np.ndarray,
    average_power_linear: np.ndarray,
    average_power_db: np.ndarray,
    noise_floor_db: float,
    settings: DetectionSettings,
) -> ChannelCenterScore:
    """Center score for one channel from its core region.

    ``center_score = max(core_median_excess_db, 0) * core_active_fraction``

    * The median needs more than half of the core to be raised, so it
      measures broadband energy and ignores narrow spikes.
    * The active fraction measures how much of the core is filled. A signal
      centered on this channel fills the core; a signal centered 5 or 10 MHz
      away fills only part of it, so the score peaks at the best-matching
      center.
    * The unit is dB. A narrow spike changes neither factor noticeably.
    """
    half = settings.core_half_bandwidth_hz
    core_start = channel.center_frequency_hz - half
    core_stop = channel.center_frequency_hz + half
    covered = (
        (frequencies_hz >= core_start)
        & (frequencies_hz <= core_stop)
        & np.isfinite(average_power_db)
    )
    core_bins = int(np.count_nonzero(covered))

    if core_bins == 0:
        nan = float("nan")
        return ChannelCenterScore(
            channel.number, channel.center_frequency_hz, core_start, core_stop,
            0, nan, nan, nan, nan, nan, note="no covered bins",
        )

    core_db = average_power_db[covered]
    core_median_db = float(np.median(core_db))
    core_excess_db = core_median_db - noise_floor_db
    active_threshold_db = noise_floor_db + settings.active_bin_margin_db
    core_active_fraction = float(np.count_nonzero(core_db >= active_threshold_db) / core_bins)

    return ChannelCenterScore(
        channel_number=channel.number,
        center_frequency_hz=channel.center_frequency_hz,
        core_start_hz=core_start,
        core_stop_hz=core_stop,
        core_valid_bins=core_bins,
        core_mean_power_db=float(power_to_db(np.mean(average_power_linear[covered]))),
        core_median_power_db=core_median_db,
        core_median_excess_db=core_excess_db,
        core_active_fraction=core_active_fraction,
        center_score=max(core_excess_db, 0.0) * core_active_fraction,
    )


def select_center_candidates(
    scores: Sequence[ChannelCenterScore],
    settings: DetectionSettings,
) -> list[ChannelCenterScore]:
    """Local-maximum test, score threshold and non-maximum suppression.

    1. Local maximum: score > 0 and >= the score of every channel whose
       number is within ``candidate_neighbor_channels`` (channels without
       data count as lowest).
    2. Threshold: score >= ``min_center_score``.
    3. Non-maximum suppression: go through the remaining channels from the
       highest score down; accept one unless it is closer than
       ``min_candidate_separation_hz`` to an already accepted candidate.

    Returns:
        New score objects with ``is_local_maximum``, ``center_candidate`` and
        ``note`` filled in, in the input order.
    """
    by_number = {s.channel_number: s for s in scores}

    def value(score: ChannelCenterScore) -> float:
        return -np.inf if np.isnan(score.center_score) else score.center_score

    local_max: dict[int, bool] = {}
    for s in scores:
        neighbours = [
            by_number[n]
            for n in range(
                s.channel_number - settings.candidate_neighbor_channels,
                s.channel_number + settings.candidate_neighbor_channels + 1,
            )
            if n != s.channel_number and n in by_number
        ]
        # A zero score (nothing above the floor) is never a meaningful maximum.
        local_max[s.channel_number] = value(s) > 0 and all(
            value(s) >= value(n) for n in neighbours
        )

    notes: dict[int, str] = {}
    accepted: list[ChannelCenterScore] = []
    ranked = sorted(
        (s for s in scores if local_max[s.channel_number]),
        key=lambda s: (-s.center_score, s.channel_number),
    )
    for s in ranked:
        if s.center_score < settings.min_center_score:
            notes[s.channel_number] = "score below minimum"
            continue
        stronger = next(
            (
                a for a in accepted
                if abs(a.center_frequency_hz - s.center_frequency_hz)
                < settings.min_candidate_separation_hz
            ),
            None,
        )
        if stronger is not None:
            notes[s.channel_number] = f"suppressed by CH{stronger.channel_number}"
            continue
        accepted.append(s)

    accepted_numbers = {a.channel_number for a in accepted}
    return [
        ChannelCenterScore(
            **{
                **s.__dict__,
                "is_local_maximum": local_max[s.channel_number],
                "center_candidate": s.channel_number in accepted_numbers,
                "note": notes.get(s.channel_number, s.note),
            }
        )
        for s in scores
    ]


# -- both layers ----------------------------------------------------------------


def detect_channel_occupancy(
    frequencies_hz: np.ndarray,
    average_power_linear: np.ndarray,
    average_power_db: np.ndarray,
    channels: Optional[Sequence[WiFiChannel]] = None,
    settings: Optional[DetectionSettings] = None,
) -> ChannelOccupancyReport:
    """Estimate the noise floor, then run both analysis layers.

    Args:
        frequencies_hz: Ascending frequency of each spectrum bin.
        average_power_linear: Stitched average power, linear, NaN if uncovered.
        average_power_db: The same spectrum in relative dB.
        channels: Channels to analyse. Defaults to config.WIFI_CHANNELS_HZ.
        settings: Detector thresholds. Defaults to the config.py values.

    Raises:
        ValueError: if the spectrum arrays are inconsistent or fully uncovered.
    """
    frequencies, linear, power_db = _validate_spectrum(
        frequencies_hz, average_power_linear, average_power_db
    )
    channels = wifi_channels() if channels is None else list(channels)
    settings = DetectionSettings() if settings is None else settings

    noise_floor_db = estimate_noise_floor_db(power_db, settings.noise_floor_percentile)
    results = [
        analyze_channel(channel, frequencies, linear, power_db, noise_floor_db, settings)
        for channel in channels
    ]
    center_scores = select_center_candidates(
        [
            score_channel_center(channel, frequencies, linear, power_db, noise_floor_db, settings)
            for channel in channels
        ],
        settings,
    )
    return ChannelOccupancyReport(
        noise_floor_db=noise_floor_db,
        active_threshold_db=noise_floor_db + settings.active_bin_margin_db,
        settings=settings,
        results=results,
        center_scores=center_scores,
    )


# -- layer 3: stable (temporal) center candidates ----------------------------------


@dataclass(frozen=True)
class TemporalCandidateResult:
    """Stable center candidates from the most recent sweeps.

    Attributes:
        scores: Per-channel scores averaged over the window, with
            ``center_candidate`` / ``note`` set by the stable decision.
        support: Sweeps in the window whose own score reached the minimum.
        sweeps_used: Number of sweeps averaged (< window while warming up).
        window_sweeps: Configured window length.
        required_support: Support needed for this result.
    """

    scores: list[ChannelCenterScore]
    support: dict[int, int]
    sweeps_used: int
    window_sweeps: int
    required_support: int

    @property
    def candidates(self) -> list[int]:
        return [s.channel_number for s in self.scores if s.center_candidate]

    @property
    def warming_up(self) -> bool:
        return self.sweeps_used < self.window_sweeps

    @property
    def temporal_scores(self) -> dict[int, float]:
        return {s.channel_number: s.center_score for s in self.scores}


def average_center_scores(
    score_sets: Sequence[Sequence[ChannelCenterScore]],
) -> list[ChannelCenterScore]:
    """Average per-channel center scores of several sweeps.

    ``center_score`` is the plain mean of the per-sweep scores (NaN sweeps
    ignored). The core metrics are averaged too, for inspection only; the
    averaged score is not recomputed from them. Candidate flags are cleared.

    Raises:
        ValueError: if there are no sweeps or the channel lists differ.
    """
    if not score_sets:
        raise ValueError("No center scores to average.")
    channels = [s.channel_number for s in score_sets[0]]
    for scores in score_sets[1:]:
        if [s.channel_number for s in scores] != channels:
            raise ValueError("All sweeps must score the same channels in the same order.")

    def mean(values: Iterable[float]) -> float:
        array = np.array(list(values), dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(finite.mean()) if finite.size else float("nan")

    averaged = []
    for index, first in enumerate(score_sets[0]):
        per_sweep = [scores[index] for scores in score_sets]
        averaged.append(
            replace(
                first,
                core_valid_bins=min(s.core_valid_bins for s in per_sweep),
                core_mean_power_db=float(
                    power_to_db(mean(10 ** (s.core_mean_power_db / 10) for s in per_sweep))
                ),
                core_median_power_db=mean(s.core_median_power_db for s in per_sweep),
                core_median_excess_db=mean(s.core_median_excess_db for s in per_sweep),
                core_active_fraction=mean(s.core_active_fraction for s in per_sweep),
                center_score=mean(s.center_score for s in per_sweep),
                is_local_maximum=False,
                center_candidate=False,
                note="",
            )
        )
    return averaged


def temporal_center_candidates(
    score_sets: Sequence[Sequence[ChannelCenterScore]],
    settings: DetectionSettings,
    window_sweeps: int = config.TEMPORAL_WINDOW_SWEEPS,
    min_support: int = config.TEMPORAL_MIN_SUPPORT,
) -> TemporalCandidateResult:
    """Stable candidates from the last ``window_sweeps`` sets of center scores.

    1. Average each channel's center_score over the window.
    2. Apply :func:`select_center_candidates` (local maximum, minimum score,
       non-maximum suppression) to the averaged scores.
    3. Drop a selected channel whose own score reached ``min_center_score``
       in fewer than ``min(min_support, sweeps available)`` sweeps.
    """
    if window_sweeps < 1 or not 1 <= min_support <= window_sweeps:
        raise ValueError("Need window_sweeps >= 1 and 1 <= min_support <= window_sweeps.")
    recent = list(score_sets)[-window_sweeps:]
    averaged = average_center_scores(recent)
    required = min(min_support, len(recent))

    support = {
        s.channel_number: sum(
            1
            for scores in recent
            for other in scores
            if other.channel_number == s.channel_number
            and np.isfinite(other.center_score)
            and other.center_score >= settings.min_center_score
        )
        for s in averaged
    }

    stable = []
    for s in select_center_candidates(averaged, settings):
        if s.center_candidate and support[s.channel_number] < required:
            s = replace(
                s,
                center_candidate=False,
                note=f"support {support[s.channel_number]}/{len(recent)} < {required}",
            )
        stable.append(s)

    return TemporalCandidateResult(
        scores=stable,
        support=support,
        sweeps_used=len(recent),
        window_sweeps=window_sweeps,
        required_support=required,
    )


class TemporalCandidateTracker:
    """Keeps the center scores of the most recent sweeps.

    Typical use, once per sweep:
        report = detect_channel_occupancy(...)
        stable = tracker.update(report)
    """

    def __init__(
        self,
        window_sweeps: int = config.TEMPORAL_WINDOW_SWEEPS,
        min_support: int = config.TEMPORAL_MIN_SUPPORT,
    ) -> None:
        if window_sweeps < 1 or not 1 <= min_support <= window_sweeps:
            raise ValueError(
                "TEMPORAL_WINDOW_SWEEPS must be >= 1 and "
                "1 <= TEMPORAL_MIN_SUPPORT <= TEMPORAL_WINDOW_SWEEPS."
            )
        self.window_sweeps = int(window_sweeps)
        self.min_support = int(min_support)
        self._score_sets: deque[list[ChannelCenterScore]] = deque(maxlen=self.window_sweeps)

    def update(self, report: ChannelOccupancyReport) -> TemporalCandidateResult:
        """Add one sweep's center scores and return the current stable result."""
        self._score_sets.append(list(report.center_scores))
        return temporal_center_candidates(
            self._score_sets, report.settings, self.window_sweeps, self.min_support
        )


# -- output helpers ----------------------------------------------------------


def format_channel_table(report: ChannelOccupancyReport) -> str:
    """Terminal table: energy-occupancy columns, then center-candidate columns."""
    header = (
        f"{'CH':<3} {'Center':>8}  {'Median':>6} {'Mean':>6} {'Peak':>6} "
        f"{'Excess':>6} {'Active%':>7}  {'Energy':<8} | "
        f"{'CoreExc':>7} {'CoreAct%':>8} {'Score':>6}  Candidate"
    )
    lines = [header, "-" * (len(header) + 14)]
    for r, s in zip(report.results, report.center_scores):
        candidate = "YES" if s.center_candidate else "NO"
        if s.note:
            candidate += f" ({s.note})"
        lines.append(
            f"{r.channel_number:<3} {r.center_frequency_hz / HZ_PER_MHZ:>4.0f} MHz  "
            f"{r.median_power_db:>6.1f} {r.mean_power_db:>6.1f} {r.peak_power_db:>6.1f} "
            f"{r.power_excess_db:>6.1f} {100 * r.active_bin_fraction:>6.1f}%  "
            f"{r.status:<8} | "
            f"{s.core_median_excess_db:>7.1f} {100 * s.core_active_fraction:>7.1f}% "
            f"{s.center_score:>6.2f}  {candidate}"
        )
    return "\n".join(lines)


CSV_COLUMNS = [
    "channel",
    "center_frequency_mhz",
    "window_start_mhz",
    "window_stop_mhz",
    "valid_bins",
    "mean_power_db",
    "median_power_db",
    "peak_power_db",
    "noise_floor_db",
    "power_excess_db",
    "active_bin_fraction",
    "energy_occupied",
    "energy_status",
    "energy_decision_reason",
    "core_start_mhz",
    "core_stop_mhz",
    "core_mean_power_db",
    "core_median_power_db",
    "core_median_excess_db",
    "core_active_fraction",
    "center_score",
    "is_local_maximum",
    "center_candidate",
    "candidate_note",
    "active_bin_margin_db",
    "channel_power_margin_db",
    "min_active_bin_fraction",
    "noise_floor_percentile",
    "core_half_bandwidth_hz",
    "min_center_score",
    "candidate_neighbor_channels",
    "min_candidate_separation_hz",
    "sweep_timestamp_utc",
]


def save_occupancy_csv(
    path: Path,
    report: ChannelOccupancyReport,
    sweep_timestamp_utc: str = "",
) -> None:
    """Write one row per channel with both layers and the settings used."""
    settings = report.settings.__dict__
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r, s in zip(report.results, report.center_scores):
            writer.writerow(
                {
                    "channel": r.channel_number,
                    "center_frequency_mhz": f"{r.center_frequency_hz / HZ_PER_MHZ:.3f}",
                    "window_start_mhz": f"{r.window_start_hz / HZ_PER_MHZ:.3f}",
                    "window_stop_mhz": f"{r.window_stop_hz / HZ_PER_MHZ:.3f}",
                    "valid_bins": r.valid_bins,
                    "mean_power_db": f"{r.mean_power_db:.3f}",
                    "median_power_db": f"{r.median_power_db:.3f}",
                    "peak_power_db": f"{r.peak_power_db:.3f}",
                    "noise_floor_db": f"{r.noise_floor_db:.3f}",
                    "power_excess_db": f"{r.power_excess_db:.3f}",
                    "active_bin_fraction": f"{r.active_bin_fraction:.4f}",
                    "energy_occupied": r.energy_occupied,
                    "energy_status": r.status,
                    "energy_decision_reason": r.decision_reason,
                    "core_start_mhz": f"{s.core_start_hz / HZ_PER_MHZ:.3f}",
                    "core_stop_mhz": f"{s.core_stop_hz / HZ_PER_MHZ:.3f}",
                    "core_mean_power_db": f"{s.core_mean_power_db:.3f}",
                    "core_median_power_db": f"{s.core_median_power_db:.3f}",
                    "core_median_excess_db": f"{s.core_median_excess_db:.3f}",
                    "core_active_fraction": f"{s.core_active_fraction:.4f}",
                    "center_score": f"{s.center_score:.4f}",
                    "is_local_maximum": s.is_local_maximum,
                    "center_candidate": s.center_candidate,
                    "candidate_note": s.note,
                    "sweep_timestamp_utc": sweep_timestamp_utc,
                    **settings,
                }
            )
