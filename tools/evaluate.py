"""Offline evaluation harness: score estimates against a longer snapshot.

The synthetic generator is deterministic, so re-running it with a later cutoff
produces the same tickets but lets every occurrence day fully report.  That
gives a ground truth for any cutoff we like.  This script is a calibration
tool, not part of the app.

Usage:
    .venv/bin/python tools/evaluate.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_ops.data import load_dataset  # noqa: E402
from ticket_ops.model import estimate_channel, train_profile  # noqa: E402
from tools.generate_data import generate  # noqa: E402

SNAPSHOT = ROOT / "data" / "tickets.csv"
TRUTH_PATH = ROOT / "data" / ".truth.csv"
CUTOFF = date(2026, 11, 22)
TRUTH_CUTOFF = date(2027, 1, 10)


def truth_counts() -> dict[tuple[str, date], int]:
    """Same tickets, every report allowed to arrive (a fully matured view)."""
    generate(TRUTH_PATH, seed=20261123, report_cutoff=TRUTH_CUTOFF)
    ds = load_dataset(TRUTH_PATH, cutoff=TRUTH_CUTOFF)
    out: dict[tuple[str, date], int] = {}
    for name, ch in ds.channels.items():
        for day in ch.occurrence_days:
            if day <= CUTOFF:
                out[(name, day)] = ch.observed_total(day)
    return out


def score(smooth: float, window: int = 28, draws: int = 60) -> None:
    ds = load_dataset(SNAPSHOT, cutoff=CUTOFF)
    truth = truth_counts()
    rows = []
    for i, (name, ch) in enumerate(ds.channels.items()):
        est = estimate_channel(ch, window=window, smooth=smooth, draws=draws, seed=1000 + i, cutoff=CUTOFF)
        for d in est.days:
            if not d.matured and d.age <= 10:
                actual = truth[(name, d.day)]
                rows.append(
                    {
                        "channel": name,
                        "day": d.day,
                        "age": d.age,
                        "est": d.estimated,
                        "truth": actual,
                        "lo": d.q05,
                        "hi": d.q95,
                        "observed": d.observed,
                    }
                )
    abs_err = np.array([abs(r["est"] - r["truth"]) for r in rows])
    rel = abs_err / np.array([max(1, r["truth"]) for r in rows])
    bias = np.array([r["est"] - r["truth"] for r in rows])
    cover = np.mean([r["lo"] <= r["truth"] <= r["hi"] for r in rows])
    width = np.mean([(r["hi"] - r["lo"]) / max(1, r["truth"]) for r in rows])
    print(
        f"smooth={smooth:6.2f} window={window:3d}  MAPE={rel.mean()*100:5.2f}%  "
        f"maxAPE={rel.max()*100:5.1f}%  bias={bias.mean():+6.1f}  cover90={cover*100:5.1f}%  "
        f"width={width*100:5.1f}%"
    )
    # Age breakdown: the newest days are the hard ones.
    for age in sorted({r["age"] for r in rows}):
        sub = [r for r in rows if r["age"] == age]
        if len(sub) < 3:
            continue
        e = np.mean([abs(r["est"] - r["truth"]) / max(1, r["truth"]) for r in sub])
        print(f"    age {age:2d}: n={len(sub):3d} MAPE={e*100:5.2f}%")


if __name__ == "__main__":
    for s in (0.0, 0.05, 0.2, 1.0, 5.0):
        score(s)
