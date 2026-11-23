"""Deterministic synthetic ticket-operations dataset.

Every row is an aggregate: how many tickets from one channel that occurred on
``occurrence_date`` were reported on ``report_date``.  Reports that would have
arrived after the observation cutoff (2026-11-22) are not in the file, so the
newest occurrence days are naturally only partially reported.

The generator deliberately plants the situations the estimation page has to
surface:

* one channel (邮件) slows down shortly before the cutoff;
* one channel (电话) reports nothing at all on 2026-11-16;
* the social channel has a couple of very long delays that history has never
  covered, reported just before the cutoff.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

START = date(2026, 5, 15)
CUTOFF = date(2026, 11, 22)
SLOWDOWN_START = date(2026, 11, 10)
OUTAGE_REPORT_DATE = date(2026, 11, 16)
OUTAGE_CHANNEL = "电话"
LONG_DELAYS = {"社交媒体": {date(2026, 10, 30): 4, date(2026, 11, 2): 2}}

CHANNELS = {
    "在线客服": {
        "base": 172.0,
        "weekday": [1.08, 1.04, 1.00, 1.00, 1.05, 0.82, 0.76],
        "trend": 0.0006,
        "delays": [0.61, 0.22, 0.085, 0.04, 0.025, 0.01, 0.005, 0.003, 0.002],
        "dispersion": 0.10,
    },
    "电话": {
        "base": 88.0,
        "weekday": [1.22, 1.10, 1.02, 1.00, 1.04, 0.55, 0.38],
        "trend": -0.0002,
        "delays": [0.44, 0.29, 0.14, 0.07, 0.035, 0.015, 0.007, 0.003],
        "dispersion": 0.12,
    },
    "邮件": {
        "base": 118.0,
        "weekday": [1.12, 1.06, 1.00, 0.98, 1.02, 0.88, 0.84],
        "trend": 0.0004,
        "delays": [0.10, 0.18, 0.20, 0.17, 0.13, 0.09, 0.06, 0.035, 0.02, 0.01, 0.005],
        "dispersion": 0.09,
    },
    "自助服务": {
        "base": 214.0,
        "weekday": [1.06, 1.03, 1.00, 1.00, 1.04, 0.90, 0.87],
        "trend": 0.0011,
        "delays": [0.71, 0.185, 0.06, 0.025, 0.012, 0.006, 0.002],
        "dispersion": 0.08,
    },
    "社交媒体": {
        "base": 58.0,
        "weekday": [1.10, 1.04, 1.00, 0.98, 1.06, 0.92, 0.90],
        "trend": 0.0005,
        "delays": [0.24, 0.215, 0.175, 0.13, 0.095, 0.06, 0.04, 0.025, 0.012, 0.006, 0.002],
        "dispersion": 0.15,
    },
    "现场": {
        "base": 33.0,
        "weekday": [1.14, 1.08, 1.02, 1.00, 1.03, 0.72, 0.55],
        "trend": 0.0002,
        "delays": [0.55, 0.29, 0.10, 0.035, 0.015, 0.006, 0.004],
        "dispersion": 0.14,
    },
}

# 邮件 slows down shortly before the cutoff: same-day reporting collapses and
# the mass moves a few days later.
EMAIL_SLOW_DELAYS = [0.035, 0.075, 0.13, 0.165, 0.165, 0.145, 0.11, 0.075, 0.045, 0.03, 0.025]


def _day_count(rng: random.Random, cfg: dict, day: date) -> int:
    mean = cfg["base"] * cfg["weekday"][day.weekday()] * math.exp(cfg["trend"] * (day - START).days)
    shock = math.exp(rng.gauss(0.0, cfg["dispersion"]))
    return max(0, int(round(rng.gauss(mean * shock, math.sqrt(mean * shock)))))


def _delay_counts(rng: random.Random, channel: str, cfg: dict, day: date, n: int) -> list[int]:
    weights = EMAIL_SLOW_DELAYS if channel == "邮件" and day >= SLOWDOWN_START else cfg["delays"]
    counts = list(rng.choices(range(len(weights)), weights=weights, k=n))
    # A handful of unusually slow tickets that history has never seen.
    counts.extend([22, 21, 19, 18, 17, 16][: LONG_DELAYS.get(channel, {}).get(day, 0)])
    return counts


def generate(
    path: Path,
    seed: int = 20261123,
    *,
    last_occurrence: date = CUTOFF,
    report_cutoff: date = CUTOFF,
) -> None:
    """Write the aggregate CSV.

    ``last_occurrence`` is the final occurrence day generated; ``report_cutoff``
    is the latest report date included.  Running with a later ``report_cutoff``
    yields the *matured* view of the same tickets, which the evaluation harness
    uses as ground truth.
    """
    rng = random.Random(seed)
    agg: dict[tuple[str, str, str], int] = defaultdict(int)
    day = START
    while day <= last_occurrence:
        for channel, cfg in CHANNELS.items():
            for delay in _delay_counts(rng, channel, cfg, day, _day_count(rng, cfg, day)):
                report = day + timedelta(days=delay)
                if report > report_cutoff:
                    continue  # not visible as of the snapshot
                if channel == OUTAGE_CHANNEL and report == OUTAGE_REPORT_DATE:
                    continue  # batch report never arrived that day
                agg[(channel, day.isoformat(), report.isoformat())] += 1
        day += timedelta(days=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "occurrence_date", "report_date", "count"])
        for (channel, occ, rep), count in sorted(agg.items()):
            writer.writerow([channel, occ, rep, count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    generate(args.out, args.seed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
