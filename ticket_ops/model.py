"""Per-channel reporting-delay model and non-negative back-projection.

The unit of analysis is an *occurrence day* with a *reporting delay* in days.
The input aggregates tickets by (channel, occurrence_date, report_date, count);
everything else is derived here.

Per channel (channels are never pooled into a shared delay profile):

1.  Delay training only accepts occurrence days that are at least ``KMAX`` days
    old, i.e. fully exposed to every delay the model can represent.  The
    horizon ``H`` is one past the longest delay seen on those days (capped at
    ``KMAX``).
2.  The delay distribution ``p[0..H-1]`` is the empirical distribution over the
    most recent ``TRAIN_LOOKBACK`` matured occurrence days.  Immature days are
    refused entry: a fresh backlog cannot contaminate the delay profile.
3.  For the trailing ``W`` occurrence days, the unreported remainder is
    recovered by non-negative least squares.  Tickets already observed are
    subtracted as a hard lower bound, so both the unknown and the constraint
    are non-negative quantities and no solution is clipped after the fact.  An
    optional second-difference penalty on the occurrence curve controls
    smoothness; the residual of the original model is reported untouched.
4.  Observed reports with delay ``>= H`` cannot be represented by ``p``.  They
    are removed from the convolution target, credited to their occurrence day
    as observed tickets, and surfaced as a coverage warning: those days can
    only be bounded from below.

Uncertainty comes from a seeded bootstrap that resamples the matured training
days and the observed report counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .data import ChannelData, Dataset, load_dataset

KMAX = 30  # longest delay the model can represent, in days
TRAIN_LOOKBACK = 91  # matured occurrence days used for the delay profile
MIN_TRAIN_DAYS = 14  # fewer than this -> channel cannot be estimated
MIN_TRAIN_TICKETS = 300
RECENT_MATURED = 14  # most recent matured days used for the slowdown check
DEFAULT_WINDOW = 28
DEFAULT_DRAWS = 200
DEFAULT_SMOOTH = 0.05
SMOOTH_MAX = 20.0
SEED = 20261123

# Warning thresholds (tuned against the bundled synthetic data)
TV_SLOWDOWN = 0.18  # truncation-adjusted TV distance for recent immature days
SAME_DAY_DROP = 0.35  # pooled same-day share versus the profile
MIN_DRIFT_TICKETS = 15  # minimum visible tickets before judging a recent day
OUTAGE_FRACTION = 0.25
OUTAGE_MIN_BASELINE = 20
WIDE_INTERVAL = 0.60
OUTAGE_UPPER_SLACK = 0.35  # extra upper-bound room for days an outage may have hidden


@dataclass
class DelayProfile:
    horizon: int
    probs: np.ndarray
    max_delay: int
    capped: bool
    train_days: int
    train_tickets: int
    recent_tv: float
    recent_same_day_ratio: float
    bootstrap_tv95: float = 0.0

    @property
    def mean_delay(self) -> float:
        return float(np.dot(np.arange(self.horizon), self.probs))


@dataclass
class DayEstimate:
    day: date
    age: int
    observed: int
    estimated: float
    q05: float
    q25: float
    q75: float
    q95: float
    unreported: float
    observed_in_horizon: int
    long_observed: int
    matured: bool
    lower_bound: bool
    contribution: list[float] = field(default_factory=list)
    observed_by_delay: list[int] = field(default_factory=list)
    draws: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)


@dataclass
class ChannelEstimate:
    channel: str
    cutoff: date
    status: str  # "ok" | "insufficient"
    profile: DelayProfile | None
    days: list[DayEstimate]
    observed_reports: dict[str, int]
    predicted_reports: dict[str, float]
    rmse: float
    warnings: list[dict]
    note: str = ""


def _day_range(start: date, end: date) -> list[date]:
    out = []
    day = start
    while day <= end:
        out.append(day)
        day += timedelta(days=1)
    return out


def _matured_days(channel: ChannelData, cutoff: date, age: int) -> list[date]:
    return [d for d in channel.occurrence_days if (cutoff - d).days >= age]


def delay_histogram(channel: ChannelData, days: list[date], horizon: int) -> np.ndarray:
    hist = np.zeros(horizon, dtype=float)
    for day in days:
        for delay, count in channel.delay_counts.get(day, {}).items():
            if 0 <= delay < horizon:
                hist[delay] += count
    return hist


def _long_observed(channel: ChannelData, day: date, horizon: int) -> int:
    return sum(c for delay, c in channel.delay_counts.get(day, {}).items() if delay >= horizon)


def _observed_in_horizon(channel: ChannelData, day: date, horizon: int) -> int:
    return sum(c for delay, c in channel.delay_counts.get(day, {}).items() if 0 <= delay < horizon)


def train_profile(
    channel: ChannelData,
    cutoff: date,
    *,
    lookback: int = TRAIN_LOOKBACK,
    kmax: int = KMAX,
) -> DelayProfile | None:
    """Empirical delay distribution using matured occurrence days only."""
    matured = _matured_days(channel, cutoff, kmax)
    if len(matured) < MIN_TRAIN_DAYS:
        return None
    train_days = matured[-lookback:]
    hist = delay_histogram(channel, train_days, kmax)
    total = hist.sum()
    if total < MIN_TRAIN_TICKETS:
        return None
    nonzero = np.flatnonzero(hist)
    max_delay = int(nonzero[-1]) if len(nonzero) else 0
    capped = max_delay >= kmax - 1
    horizon = min(kmax, max_delay + 1)
    probs = hist[:horizon] / hist[:horizon].sum()

    recent = matured[-RECENT_MATURED:]
    recent_hist = delay_histogram(channel, recent, horizon)
    recent_total = recent_hist.sum()
    recent_tv = 0.0
    same_day_ratio = float("nan")
    if recent_total >= 200:
        recent_p = recent_hist / recent_total
        recent_tv = float(0.5 * np.abs(recent_p - probs).sum())
        if probs[0] > 0:
            same_day_ratio = float(recent_p[0] / probs[0])
    return DelayProfile(
        horizon=horizon,
        probs=probs,
        max_delay=max_delay,
        capped=capped,
        train_days=len(train_days),
        train_tickets=int(total),
        recent_tv=recent_tv,
        recent_same_day_ratio=same_day_ratio,
    )


class _Problem:
    """The convolution system for one channel, window and delay profile."""

    def __init__(self, channel: ChannelData, profile: DelayProfile, cutoff: date, window: int):
        self.channel = channel
        self.profile = profile
        self.cutoff = cutoff
        self.h = profile.horizon
        self.days = [d for d in _day_range(max(channel.first_occurrence, cutoff - timedelta(days=window - 1)), cutoff)]
        self.free = [d for d in self.days if (cutoff - d).days < self.h - 1]
        self.fixed = [d for d in self.days if (cutoff - d).days >= self.h - 1]
        self.free_col = {d: i for i, d in enumerate(self.free)}
        self.fixed_col = {d: i for i, d in enumerate(self.fixed)}
        self.report_dates = _day_range(self.days[0], cutoff)
        self.observed_reports = np.array([channel.report_counts.get(t, 0) for t in self.report_dates], dtype=float)
        # Reports contributed by occurrences before the window, and by
        # in-window occurrences whose delay the profile cannot represent.
        self.old_contrib = np.zeros(len(self.report_dates))
        self.long_contrib = np.zeros(len(self.report_dates))
        report_index = {t: i for i, t in enumerate(self.report_dates)}
        for row in channel.rows:
            idx = report_index.get(row.report)
            if idx is None:
                continue
            if row.occurrence < self.days[0]:
                self.old_contrib[idx] += row.count
            elif row.occurrence in self.free_col or row.occurrence in self.fixed_col:
                if row.delay >= self.h:
                    self.long_contrib[idx] += row.count
        self.target = self.observed_reports - self.old_contrib - self.long_contrib
        self.obs_free = np.array([_observed_in_horizon(channel, d, self.h) for d in self.free], dtype=float)
        self.fixed_values = np.array(
            [channel.observed_total(d) - _long_observed(channel, d, self.h) for d in self.fixed], dtype=float
        )
        self.long_by_day = {d: _long_observed(channel, d, self.h) for d in self.days}
        # A week-scale reference for making the smoothness weight dimensionless.
        weekday_totals = [sum(channel.delay_counts.get(d, {}).values()) for d in self.days]
        positives = [v for v in weekday_totals if v > 0]
        self.scale = float(np.median(positives)) if positives else 1.0
        self._build_penalty()

    # -- matrices ---------------------------------------------------------
    def matrix(self, probs: np.ndarray) -> np.ndarray:
        a = np.zeros((len(self.report_dates), len(self.days)))
        for j, d in enumerate(self.days):
            for i, t in enumerate(self.report_dates):
                delay = (t - d).days
                if 0 <= delay < self.h:
                    a[i, j] = probs[delay]
        return a

    def split(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        free_cols = [self.days.index(d) for d in self.free]
        fixed_cols = [self.days.index(d) for d in self.fixed]
        a_free = a[:, free_cols] if free_cols else np.zeros((a.shape[0], 0))
        a_fixed = a[:, fixed_cols] if fixed_cols else np.zeros((a.shape[0], 0))
        return a_free, a_fixed

    def _build_penalty(self) -> None:
        n = len(self.days)
        self.second_free = None
        self.second_fixed = None
        if n < 3:
            return
        # First differences of the occurrence curve.  Penalising the *second*
        # difference would also penalise the weekly level shifts the curve
        # legitimately has, which biases the estimate upward.
        first = np.zeros((n - 1, n))
        for i in range(n - 1):
            first[i, i : i + 2] = (-1.0, 1.0)
        self.second_free = first[:, [self.days.index(d) for d in self.free]] if self.free else None
        self.second_fixed = first[:, [self.days.index(d) for d in self.fixed]] if self.fixed else None

    def penalty_terms(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Restate ``R z`` in terms of the free unknowns ``w``.

        ``z = obs_free + w`` on free days and ``fixed_values`` on matured days,
        so ``R z = R_free w + R_free obs_free + R_fixed z_fixed``.
        """
        if self.second_free is None:
            return None, None
        target = self.second_free @ self.obs_free
        if self.second_fixed is not None:
            target = target + self.second_fixed @ self.fixed_values
        return self.second_free, -target

    def solve(
        self, probs: np.ndarray, target: np.ndarray, smooth: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (free unknowns w, full occurrence counts z, RMSE of the data term).

        Rows are weighted by the inverse Poisson standard deviation of their
        target (``sqrt(max(target, 1))``), so a quiet recent report day does not
        outvote a busy one.  The smoothness penalty is a dimensionless weight on
        the second difference of the *scaled* occurrence curve, so the slider
        means the same thing at every channel volume.  Non-negativity is
        enforced by NNLS on the unknowns; observed tickets enter as a lower
        bound, never as a post-hoc clip.
        """
        a = self.matrix(probs)
        a_free, a_fixed = self.split(a)
        if len(self.free) == 0:
            z = np.zeros(len(self.days))
            for j, d in enumerate(self.days):
                z[j] = self.fixed_values[self.fixed_col[d]]
            resid = target - a @ z
            return np.zeros(0), z, float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
        sqrt_w = 1.0 / np.sqrt(np.maximum(target, 1.0))
        rhs = target - a_free @ self.obs_free - (a_fixed @ self.fixed_values if len(self.fixed) else 0.0)
        penalty_free, penalty_target = self.penalty_terms()
        rows = [a_free * sqrt_w[:, None]]
        cols = [rhs * sqrt_w]
        if penalty_free is not None and smooth > 0:
            # Curvature of the scaled curve, in units of the typical daily
            # weight: lam=1 makes one unit of relative curvature cost about one
            # unit of Poisson residual.
            scale = max(1.0, self.scale)
            lam = float(np.sqrt(smooth))
            typical_sd = float(np.sqrt(max(float(np.median(np.maximum(target, 0.0))), 1.0)))
            rows.append(lam * penalty_free * (typical_sd / scale))
            cols.append(lam * penalty_target * (typical_sd / scale))
        stacked = np.vstack(rows)
        b = np.concatenate(cols)
        w, _ = nnls(stacked, b, maxiter=30 * max(1, stacked.shape[1]))
        z = self.expand(w)
        resid = a @ z - self.target
        rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
        return w, z, rmse

    def expand(self, w: np.ndarray) -> np.ndarray:
        z = np.zeros(len(self.days))
        for j, d in enumerate(self.days):
            if d in self.free_col:
                z[j] = self.obs_free[self.free_col[d]] + w[self.free_col[d]]
            else:
                # A matured day's true count includes any observed delays the
                # profile cannot represent, so it is a lower bound of the total.
                z[j] = max(
                    self.fixed_values[self.fixed_col[d]],
                    float(self.channel.observed_total(d)),
                )
        return z

    # -- diagnostics ------------------------------------------------------
    def progress_check(self) -> tuple[float, float]:
        """Truncation-safe drift check on days that are not yet mature.

        Immature occurrence days are right-censored: only delays ``0..age`` are
        visible.  Comparing the *conditional* distribution of the visible part
        against the profile conditioned on the same truncation is valid without
        knowing the missing tail, and it never feeds immature data back into
        delay training.

        Returns (weighted TV distance, pooled same-day ratio).  Both are
        ``nan`` when there is too little recent traffic to judge.
        """
        tvs: list[float] = []
        weights: list[float] = []
        same_day = 0.0
        seen_total = 0.0
        for d in self.days:
            age = (self.cutoff - d).days
            if age < 1 or age > self.h - 2:
                continue
            counts = self.channel.delay_counts.get(d, {})
            visible = np.array([counts.get(k, 0) for k in range(age + 1)], dtype=float)
            total = visible.sum()
            if total < MIN_DRIFT_TICKETS:
                continue
            observed_p = visible / total
            expected = self.profile.probs[: age + 1]
            expected = expected / expected.sum()
            tvs.append(float(0.5 * np.abs(observed_p - expected).sum()))
            weights.append(float(total))
            same_day += visible[0]
            seen_total += total
        if not tvs:
            return float("nan"), float("nan")
        tv = float(np.average(tvs, weights=weights))
        ratio = float((same_day / seen_total) / self.profile.probs[0]) if seen_total else float("nan")
        return tv, ratio


def _bootstrap_probs(
    channel: ChannelData, profile: DelayProfile, cutoff: date, rng: np.random.Generator
) -> np.ndarray:
    matured = _matured_days(channel, cutoff, KMAX)[-TRAIN_LOOKBACK:]
    if len(matured) < 2:
        return profile.probs
    picks = rng.integers(0, len(matured), size=len(matured))
    hist = np.zeros(profile.horizon, dtype=float)
    for idx in picks:
        for delay, count in channel.delay_counts.get(matured[int(idx)], {}).items():
            if 0 <= delay < profile.horizon:
                hist[delay] += count
    if hist.sum() <= 0:
        return profile.probs
    return hist / hist.sum()


def _warnings(
    channel: ChannelData,
    profile: DelayProfile,
    cutoff: date,
    days: list[DayEstimate],
    by_report_date: dict[date, int],
    drift: tuple[float, float],
) -> tuple[list[dict], date | None]:
    """Build channel warnings, returning the most recent suspected outage day."""
    warnings: list[dict] = []
    if profile.train_days < 30:
        warnings.append(
            {
                "kind": "history_short",
                "severity": "warning",
                "title": "成熟历史偏少",
                "detail": f"仅 {profile.train_days} 个成熟发生日（{profile.train_tickets} 单）参与延迟训练。",
            }
        )
    if profile.capped:
        warnings.append(
            {
                "kind": "horizon_capped",
                "severity": "warning",
                "title": "延迟训练触及上限",
                "detail": (
                    f"已观察到 {profile.max_delay} 天延迟；模型最多表示 {profile.horizon - 1} 天，"
                    "更长延迟无法被分解。"
                ),
            }
        )
    drift_tv, drift_ratio = drift
    slow = (not np.isnan(drift_ratio) and drift_ratio < 1 - SAME_DAY_DROP) or (
        not np.isnan(drift_tv) and drift_tv >= TV_SLOWDOWN
    )
    if slow:
        pieces = []
        if not np.isnan(drift_tv):
            pieces.append(f"截断校正后的延迟分布与训练期相差 TV={drift_tv:.2f}")
        if not np.isnan(drift_ratio):
            pieces.append(f"当日上报占比是训练期的 {drift_ratio:.0%}")
        warnings.append(
            {
                "kind": "slowdown",
                "severity": "serious",
                "title": "该渠道上报变慢",
                "detail": (
                    "最近尚未报齐的发生日：" + "，".join(pieces) + "。"
                    "延迟分布仍按成熟历史训练（未成熟批次不回流）；若变慢持续，最近几日的估计会偏低，"
                    "请以区间上界作为参考。"
                ),
            }
        )
    recent_reports = _day_range(max(cutoff - timedelta(days=13), min(by_report_date)), cutoff) if by_report_date else []
    outage_days: list[date] = []
    for report_day in recent_reports:
        actual = by_report_date.get(report_day, 0)
        baseline_values = []
        for weeks in range(1, 9):
            prev = by_report_date.get(report_day - timedelta(days=7 * weeks))
            if prev is not None:
                baseline_values.append(float(prev))
        if not baseline_values:
            continue
        baseline = float(np.median(baseline_values))
        if baseline >= OUTAGE_MIN_BASELINE and actual <= OUTAGE_FRACTION * baseline:
            outage_days.append(report_day)
            warnings.append(
                {
                    "kind": "outage",
                    "severity": "critical",
                    "title": "上报日疑似停报",
                    "detail": (
                        f"{report_day.isoformat()} 仅收到 {actual} 单上报，同星期历史中位数 {baseline:.0f} 单。"
                        "当日应到未到的上报没有进入数据；这些上报本应统计到更早的发生日，"
                        "因此相关发生日只给出下界，其上界不据此封顶。"
                    ),
                    "day": report_day.isoformat(),
                }
            )
    long_days = [d for d in days if d.long_observed > 0]
    if long_days:
        example = long_days[0]
        warnings.append(
            {
                "kind": "uncovered_delay",
                "severity": "serious",
                "title": "出现历史未覆盖的延迟",
                "detail": (
                    f"{len(long_days)} 个发生日出现了不低于 {profile.horizon} 天的延迟"
                    f"（如 {example.day.isoformat()} 有 {example.long_observed} 单）。这些延迟不在训练分布内，"
                    "对应发生日按已观察量计为下界；未成熟批次不回流训练。"
                ),
            }
        )
    wide = [
        d
        for d in days
        if not d.matured
        and d.estimated > 0
        and (d.q95 - d.q05) / max(d.estimated, 1.0) >= WIDE_INTERVAL
    ]
    if wide:
        warnings.append(
            {
                "kind": "wide_interval",
                "severity": "warning",
                "title": "估计区间偏宽",
                "detail": f"{len(wide)} 个发生日的 90% 区间超过估计值的 {WIDE_INTERVAL:.0%}，请勿当作确定值使用。",
            }
        )
    return warnings, (max(outage_days) if outage_days else None)


def estimate_channel(
    channel: ChannelData,
    *,
    window: int = DEFAULT_WINDOW,
    smooth: float = DEFAULT_SMOOTH,
    draws: int = DEFAULT_DRAWS,
    seed: int = SEED,
    cutoff: date | None = None,
) -> ChannelEstimate:
    cutoff = cutoff or channel.cutoff
    profile = train_profile(channel, cutoff)
    window_days = _day_range(max(channel.first_occurrence, cutoff - timedelta(days=window - 1)), cutoff)
    if profile is None:
        days = [
            DayEstimate(
                day=d,
                age=(cutoff - d).days,
                observed=channel.observed_total(d),
                estimated=float(channel.observed_total(d)),
                q05=float(channel.observed_total(d)),
                q25=float(channel.observed_total(d)),
                q75=float(channel.observed_total(d)),
                q95=float(channel.observed_total(d)),
                unreported=0.0,
                observed_in_horizon=channel.observed_total(d),
                long_observed=0,
                matured=False,
                lower_bound=False,
            )
            for d in window_days
        ]
        return ChannelEstimate(
            channel=channel.name,
            cutoff=cutoff,
            status="insufficient",
            profile=None,
            days=days,
            observed_reports={},
            predicted_reports={},
            rmse=float("nan"),
            warnings=[
                {
                    "kind": "no_profile",
                    "severity": "critical",
                    "title": "成熟历史不足，无法估计",
                    "detail": "该渠道缺少足够的已成熟发生日，无法训练延迟分布，仅显示已上报数量。",
                }
            ],
            note="insufficient matured history",
        )

    problem = _Problem(channel, profile, cutoff, window)
    w, z, rmse = problem.solve(profile.probs, problem.target, smooth)

    # Smoothness-sensitivity envelope: the interval must reflect the tuning
    # knob, not just sampling noise.  Re-solve at neighbouring settings.
    sensitivity_points = [0.5 * smooth, smooth, 2.0 * smooth] if smooth > 0 else [0.0]
    sens = np.vstack([problem.solve(profile.probs, problem.target, s)[1] for s in sensitivity_points])
    sens_lo = sens.min(axis=0)
    sens_hi = sens.max(axis=0)

    rng = np.random.default_rng(seed)
    draw_matrix = np.zeros((draws, len(problem.days)))
    boot_tv = np.zeros(draws)
    scale = max(1.0, float(np.mean(np.abs(problem.target)))) if len(problem.target) else 1.0
    for b in range(draws):
        p_boot = _bootstrap_probs(channel, profile, cutoff, rng)
        boot_tv[b] = float(0.5 * np.abs(p_boot - profile.probs).sum())
        target_boot = rng.poisson(np.maximum(problem.target, 0.0)).astype(float)
        _, z_boot, _ = problem.solve(p_boot, target_boot, smooth)
        draw_matrix[b] = z_boot
    profile.bootstrap_tv95 = float(np.quantile(boot_tv, 0.95)) if draws else 0.0

    estimates: list[DayEstimate] = []
    for j, d in enumerate(window_days):
        boot = draw_matrix[:, j] if draws > 0 else np.array([float(z[j])])
        est = float(z[j])
        observed = channel.observed_total(d)
        contribution = [est * float(profile.probs[k]) for k in range(profile.horizon)]
        observed_by_delay = [
            int(channel.delay_counts.get(d, {}).get(k, 0)) for k in range(profile.horizon)
        ]
        lo = min(float(np.quantile(boot, 0.05)), float(sens_lo[j]), est, float(observed))
        hi = max(float(np.quantile(boot, 0.95)), float(sens_hi[j]), est)
        estimates.append(
            DayEstimate(
                day=d,
                age=(cutoff - d).days,
                observed=observed,
                estimated=est,
                q05=lo,
                q25=min(float(np.quantile(boot, 0.25)), est),
                q75=max(float(np.quantile(boot, 0.75)), est),
                q95=hi,
                unreported=max(0.0, est - observed),
                observed_in_horizon=_observed_in_horizon(channel, d, profile.horizon),
                long_observed=problem.long_by_day[d],
                matured=d in problem.fixed_col,
                lower_bound=problem.long_by_day[d] > 0,
                contribution=contribution,
                observed_by_delay=observed_by_delay,
                draws=boot,
            )
        )

    predicted_reports: dict[str, float] = {}
    for t in _day_range(problem.days[0], cutoff + timedelta(days=profile.horizon - 1)):
        total = 0.0
        for j, d in enumerate(problem.days):
            delay = (t - d).days
            if 0 <= delay < profile.horizon:
                total += z[j] * profile.probs[delay]
        if t <= cutoff:
            total += problem.old_contrib[problem.report_dates.index(t)] if t in problem.report_dates else 0.0
        predicted_reports[t.isoformat()] = total

    by_report_date = dict(channel.report_counts)
    warnings, outage_day = _warnings(
        channel, profile, cutoff, estimates, by_report_date, problem.progress_check()
    )
    if outage_day is not None:
        # An outage hides reports that belonged to earlier occurrence days.  The
        # missing mass is unknown, so mark the affected days as lower bounds
        # instead of reporting a point estimate that pretends to be exact, and
        # widen their upper bound rather than letting the shortfall cap it.
        for day_est in estimates:
            if day_est.matured or not _within_report_reach(day_est.day, outage_day, profile.horizon):
                continue
            day_est.lower_bound = True
            day_est.q95 = max(
                day_est.q95,
                day_est.estimated * (1.0 + OUTAGE_UPPER_SLACK),
                float(day_est.observed) + day_est.unreported,
            )
    return ChannelEstimate(
        channel=channel.name,
        cutoff=cutoff,
        status="ok",
        profile=profile,
        days=estimates,
        observed_reports={t.isoformat(): int(v) for t, v in zip(problem.report_dates, problem.observed_reports)},
        predicted_reports=predicted_reports,
        rmse=rmse,
        warnings=warnings,
    )


def _within_report_reach(occurrence: date, report_day: date, h: int) -> bool:
    """Could an outage on ``report_day`` have swallowed reports for ``occurrence``?"""
    delay = (report_day - occurrence).days
    return 0 <= delay < h


def aggregate(estimates: dict[str, ChannelEstimate]) -> dict:
    """Sum quantity across channels; channels are fitted separately first."""
    usable = {k: v for k, v in estimates.items() if v.status == "ok" and v.days}
    if not usable:
        return {"days": [], "observed_reports": {}, "predicted_reports": {}}
    day_keys = sorted({d.day for est in usable.values() for d in est.days})
    days = []
    for day in day_keys:
        rows = [next(d for d in est.days if d.day == day) for est in usable.values() if any(d.day == day for d in est.days)]
        est = sum(r.estimated for r in rows)
        total_draws = np.sum([r.draws for r in rows], axis=0)
        days.append(
            {
                "day": day.isoformat(),
                "age": rows[0].age,
                "observed": sum(r.observed for r in rows),
                "estimated": est,
                "q05": float(np.quantile(total_draws, 0.05)),
                "q25": float(np.quantile(total_draws, 0.25)),
                "q75": float(np.quantile(total_draws, 0.75)),
                "q95": float(np.quantile(total_draws, 0.95)),
                "unreported": sum(r.unreported for r in rows),
                "matured": all(r.matured for r in rows),
                "lower_bound": any(r.lower_bound for r in rows),
            }
        )
    observed_reports: dict[str, int] = {}
    predicted_reports: dict[str, float] = {}
    for est in usable.values():
        for key, value in est.observed_reports.items():
            observed_reports[key] = observed_reports.get(key, 0) + value
        for key, value in est.predicted_reports.items():
            predicted_reports[key] = predicted_reports.get(key, 0.0) + value
    return {
        "days": days,
        "observed_reports": dict(sorted(observed_reports.items())),
        "predicted_reports": dict(sorted(predicted_reports.items())),
    }


def estimate_dataset(
    dataset: Dataset,
    *,
    window: int = DEFAULT_WINDOW,
    smooth: float = DEFAULT_SMOOTH,
    draws: int = DEFAULT_DRAWS,
    seed: int = SEED,
) -> dict[str, ChannelEstimate]:
    return {
        name: estimate_channel(
            channel, window=window, smooth=smooth, draws=draws, seed=seed + i, cutoff=dataset.cutoff
        )
        for i, (name, channel) in enumerate(dataset.channels.items())
    }


def replay(
    data_path: str | Path,
    *,
    current_cutoff: date | None = None,
    replay_cutoffs: list[date] | None = None,
    window: int = DEFAULT_WINDOW,
    smooth: float = DEFAULT_SMOOTH,
    draws: int = 60,
    seed: int = SEED,
) -> dict:
    """Replay earlier cutoffs and score them against the now-matured truth.

    Each replay re-reads the file with every report after that cutoff
    physically excluded, so it can only use history that was matured at the
    time.  The comparison value is the count those occurrence days eventually
    reached by the current cutoff, restricted to days that are matured now.
    """
    now = load_dataset(data_path, cutoff=current_cutoff)
    truth: dict[str, dict[date, float]] = {}
    matured_now: dict[str, set[date]] = {}
    for name, channel in now.channels.items():
        profile = train_profile(channel, now.cutoff)
        if profile is None:
            continue
        matured = {d for d in channel.occurrence_days if (now.cutoff - d).days >= profile.horizon - 1}
        matured_now[name] = matured
        truth[name] = {d: float(channel.observed_total(d)) for d in matured}

    if replay_cutoffs is None:
        replay_cutoffs = [now.cutoff - timedelta(days=k) for k in (28, 21, 14, 7)]

    replays = []
    for offset, replay_cutoff in enumerate(replay_cutoffs):
        if replay_cutoff >= now.cutoff:
            continue
        past = load_dataset(data_path, cutoff=replay_cutoff)
        channels_out = {}
        for name, channel in past.channels.items():
            if name not in matured_now:
                continue
            profile = train_profile(channel, replay_cutoff)
            if profile is None:
                continue
            est = estimate_channel(
                channel,
                window=window,
                smooth=smooth,
                draws=draws,
                seed=seed + 100 * offset,
                cutoff=replay_cutoff,
            )
            pairs = []
            for day_est in est.days:
                if day_est.day not in matured_now[name]:
                    continue
                observed_then = float(channel.observed_total(day_est.day))
                pairs.append(
                    {
                        "day": day_est.day.isoformat(),
                        "age_then": day_est.age,
                        # Not matured at the replay cutoff: this is the case the
                        # estimate is supposed to solve, and the one worth scoring.
                        "immature_then": not day_est.matured,
                        "estimated": day_est.estimated,
                        "q05": day_est.q05,
                        "q95": day_est.q95,
                        "observed_then": observed_then,
                        "matured_now": truth[name][day_est.day],
                        "later_arrived": truth[name][day_est.day] - observed_then,
                    }
                )
            scored = [p for p in pairs if p["immature_then"]]
            if not scored:
                continue
            abs_err = [abs(p["matured_now"] - p["estimated"]) for p in scored]
            bias = [p["estimated"] - p["matured_now"] for p in scored]
            coverage = [p for p in scored if p["q05"] <= p["matured_now"] <= p["q95"]]
            channels_out[name] = {
                "profile_horizon": profile.horizon,
                "train_days": profile.train_days,
                "pairs": pairs,
                "scored_days": len(scored),
                "mape": float(np.mean([e / max(1.0, p["matured_now"]) for e, p in zip(abs_err, scored)])),
                "bias": float(np.mean(bias)),
                "coverage90": len(coverage) / len(scored),
                "naive_mape": float(
                    np.mean([abs(p["matured_now"] - p["observed_then"]) / max(1.0, p["matured_now"]) for p in scored])
                ),
            }
        replays.append({"cutoff": replay_cutoff.isoformat(), "channels": channels_out})

    summary = {}
    for name in now.channels:
        rows = [r["channels"][name] for r in replays if name in r["channels"]]
        if not rows:
            continue
        summary[name] = {
            "mape": float(np.mean([r["mape"] for r in rows])),
            "bias": float(np.mean([r["bias"] for r in rows])),
            "coverage90": float(np.mean([r["coverage90"] for r in rows])),
            "naive_mape": float(np.mean([r["naive_mape"] for r in rows])),
        }
    return {"current_cutoff": now.cutoff.isoformat(), "replays": replays, "summary": summary}
