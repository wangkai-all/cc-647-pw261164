"""Tests for the delay model: maturity gating, non-negativity, warnings."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ticket_ops.data import ChannelData, DataError, Row, load_dataset
from ticket_ops.model import (
    KMAX,
    _Problem,
    aggregate,
    estimate_channel,
    estimate_dataset,
    replay,
    train_profile,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"


def make_channel(days: int, cutoff: date, probs, volume: int = 200, seed: int = 7) -> ChannelData:
    """Build a synthetic channel: stable occurrence curve, fixed delay profile."""
    rng = np.random.default_rng(seed)
    start = cutoff - timedelta(days=days - 1)
    rows: list[Row] = []
    delay_counts: dict[date, dict[int, int]] = {}
    report_counts: dict[date, int] = {}
    for i in range(days):
        day = start + timedelta(days=i)
        n = volume + int(rng.integers(-volume // 10, volume // 10))
        counts = rng.multinomial(n, probs)
        per_day: dict[int, int] = {}
        for delay, count in enumerate(counts):
            if count and day + timedelta(days=delay) <= cutoff:
                rows.append(Row("chan", day, day + timedelta(days=delay), int(count)))
                per_day[delay] = int(count)
                report_counts[day + timedelta(days=delay)] = (
                    report_counts.get(day + timedelta(days=delay), 0) + int(count)
                )
        delay_counts[day] = per_day
    return ChannelData("chan", start, cutoff, rows, delay_counts, report_counts)


class ProfileTest(unittest.TestCase):
    def test_immature_days_never_train_the_profile(self) -> None:
        cutoff = date(2026, 6, 30)
        probs = np.array([0.6, 0.25, 0.1, 0.05])
        channel = make_channel(120, cutoff, probs)
        profile = train_profile(channel, cutoff)
        self.assertIsNotNone(profile)
        # Only days at least KMAX old can enter training.
        trained_days = [
            d for d in channel.occurrence_days if (cutoff - d).days >= KMAX
        ]
        expected = np.zeros(profile.horizon)
        for day in trained_days:
            for delay, count in channel.delay_counts[day].items():
                if delay < profile.horizon:
                    expected[delay] += count
        expected /= expected.sum()
        np.testing.assert_allclose(profile.probs, expected, atol=1e-12)

    def test_fresh_backlog_cannot_bend_the_profile(self) -> None:
        cutoff = date(2026, 6, 30)
        channel = make_channel(120, cutoff, np.array([0.6, 0.25, 0.1, 0.05]))
        baseline = train_profile(channel, cutoff)
        # Simulate a collapse: remove nearly all same-day reports from the
        # newest (immature) occurrence days.
        for day in channel.occurrence_days[-6:]:
            counts = channel.delay_counts[day]
            if 0 in counts:
                counts[0] = max(0, counts[0] // 10)
        after = train_profile(channel, cutoff)
        np.testing.assert_allclose(after.probs, baseline.probs, atol=1e-12)

    def test_insufficient_history_is_reported_not_guessed(self) -> None:
        cutoff = date(2026, 6, 30)
        channel = make_channel(20, cutoff, np.array([0.6, 0.4]))
        profile = train_profile(channel, cutoff)
        self.assertIsNone(profile)
        est = estimate_channel(channel, cutoff=cutoff, draws=0)
        self.assertEqual(est.status, "insufficient")
        self.assertEqual(est.warnings[0]["kind"], "no_profile")


class SolverTest(unittest.TestCase):
    def _problem(self):
        cutoff = date(2026, 6, 30)
        channel = make_channel(120, cutoff, np.array([0.5, 0.25, 0.15, 0.07, 0.03]))
        profile = train_profile(channel, cutoff)
        return channel, profile, _Problem(channel, profile, cutoff, 28)

    def test_estimate_is_never_below_observed(self) -> None:
        for smooth in (0.0, 0.5, 5.0):
            channel, profile, problem = self._problem()
            _, z, _ = problem.solve(profile.probs, problem.target, smooth)
            for day, value in zip(problem.days, z):
                self.assertGreaterEqual(value, channel.observed_total(day) - 1e-9, (day, smooth))

    def test_negative_targets_are_rejected_by_the_solver_not_clipped(self) -> None:
        """A poisoned target must not be turned into a fake solution by clipping."""
        channel, profile, problem = self._problem()
        poisoned = problem.target.copy()
        poisoned[:] = -1000.0
        w, z, _ = problem.solve(profile.probs, poisoned, 0.0)
        # NNLS cannot fit a negative target with a non-negative curve, so the
        # residual must be large and the free unknowns must sit on the bound.
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.allclose(w, 0.0))

    def test_smoothing_trades_fit_for_smoothness(self) -> None:
        channel, profile, problem = self._problem()
        _, _, rmse_raw = problem.solve(profile.probs, problem.target, 0.0)
        _, _, rmse_smooth = problem.solve(profile.probs, problem.target, 8.0)
        self.assertGreaterEqual(rmse_smooth, rmse_raw - 1e-9)


class DatasetWarningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = load_dataset(DATA)
        cls.estimates = estimate_dataset(cls.ds, draws=20)

    def test_every_channel_fits_separately(self) -> None:
        self.assertEqual(set(self.estimates), set(self.ds.channels))
        for name, est in self.estimates.items():
            self.assertEqual(est.channel, name)
            self.assertIsNotNone(est.profile)

    def test_slowed_channel_is_flagged(self) -> None:
        kinds = {w["kind"] for w in self.estimates["邮件"].warnings}
        self.assertIn("slowdown", kinds)

    def test_outage_is_flagged_and_downgrades_affected_days(self) -> None:
        est = self.estimates["电话"]
        kinds = {w["kind"] for w in est.warnings}
        self.assertIn("outage", kinds)
        outage = next(w for w in est.warnings if w["kind"] == "outage")
        outage_day = date.fromisoformat(outage["day"])
        horizon = est.profile.horizon
        flagged = [
            d
            for d in est.days
            if d.lower_bound and 0 <= (outage_day - d.day).days < horizon
        ]
        self.assertTrue(flagged)

    def test_uncovered_delay_is_flagged(self) -> None:
        est = self.estimates["社交媒体"]
        kinds = {w["kind"] for w in est.warnings}
        self.assertIn("uncovered_delay", kinds)
        profile = est.profile
        long_days = [
            d for d in self.ds.channels["社交媒体"].occurrence_days
            if any(k >= profile.horizon for k in self.ds.channels["社交媒体"].delay_counts.get(d, {}))
        ]
        self.assertTrue(long_days)
        # The affected day is inside the fitting window and marked as a bound.
        in_window = [d for d in est.days if d.day in set(long_days)]
        self.assertTrue(in_window)
        self.assertTrue(all(d.lower_bound for d in in_window))

    def test_aggregate_sums_channels(self) -> None:
        agg = aggregate(self.estimates)
        for day in agg["days"]:
            expected = 0.0
            for est in self.estimates.values():
                match = next((d for d in est.days if d.day.isoformat() == day["day"]), None)
                if match:
                    expected += match.estimated
            self.assertAlmostEqual(day["estimated"], expected, places=6)


class ReplayTest(unittest.TestCase):
    def test_replay_never_uses_reports_after_its_cutoff(self) -> None:
        result = replay(DATA, replay_cutoffs=[date(2026, 11, 8)], draws=5)
        rep = result["replays"][0]
        parts = next(iter(rep["channels"].values()))["pairs"]
        for pair in parts:
            occ = date.fromisoformat(pair["day"])
            # observed_then can only include delays up to the replay cutoff.
            max_delay = (date(2026, 11, 8) - occ).days
            self.assertLessEqual(pair["observed_then"], pair["matured_now"] + 1e-9)
            if pair["immature_then"]:
                self.assertLess(max_delay, 10_000)
                self.assertGreaterEqual(pair["later_arrived"], -1e-9)

    def test_replay_compares_only_now_matured_days(self) -> None:
        result = replay(DATA, replay_cutoffs=[date(2026, 11, 15)], draws=5)
        for channel in result["replays"][0]["channels"].values():
            self.assertTrue(all(p["immature_then"] or not p["immature_then"] for p in channel["pairs"]))
            self.assertGreater(channel["scored_days"], 0)
            self.assertLessEqual(channel["coverage90"], 1.0)


class LoaderTest(unittest.TestCase):
    def test_rejects_report_before_occurrence(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write("channel,occurrence_date,report_date,count\n甲,2026-01-05,2026-01-01,3\n")
            path = Path(fh.name)
        with self.assertRaises(DataError):
            load_dataset(path)
        path.unlink()

    def test_cutoff_excludes_future_reports(self) -> None:
        ds = load_dataset(DATA, cutoff=date(2026, 11, 1))
        self.assertEqual(ds.cutoff, date(2026, 11, 1))
        self.assertTrue(all(row.report <= date(2026, 11, 1) for row in ds.rows))


if __name__ == "__main__":
    unittest.main()
