"""Loading and validation for the ticket reporting dataset.

Input is a CSV of aggregates with the columns::

    channel, occurrence_date, report_date, count

Each row says: ``count`` tickets from ``channel`` that *occurred* on
``occurrence_date`` were *reported* on ``report_date``.  Rows that would have
been reported after the observation cutoff simply are not in the file.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REQUIRED_COLUMNS = ("channel", "occurrence_date", "report_date", "count")


class DataError(ValueError):
    """The input dataset is unusable."""


@dataclass(frozen=True)
class Row:
    channel: str
    occurrence: date
    report: date
    count: int

    @property
    def delay(self) -> int:
        return (self.report - self.occurrence).days


@dataclass
class ChannelData:
    """All observed reports for one channel, indexed for the model."""

    name: str
    first_occurrence: date
    cutoff: date
    rows: list[Row]
    # (occurrence date, delay) -> observed count, aggregated over report dates.
    delay_counts: dict[date, dict[int, int]]
    # report date -> observed count from any occurrence date.
    report_counts: dict[date, int]
    # report date -> observed count from occurrences on/after ``window_start``
    # is filled in by the model; these raw fields are enough to derive it.

    @property
    def occurrence_days(self) -> list[date]:
        out = []
        day = self.first_occurrence
        while day <= self.cutoff:
            out.append(day)
            day += timedelta(days=1)
        return out

    def counts_for_day(self, day: date) -> dict[int, int]:
        return self.delay_counts.get(day, {})

    def observed_total(self, day: date) -> int:
        """Tickets from ``day`` seen on or before the cutoff."""
        return sum(self.delay_counts.get(day, {}).values())

    def delay_samples(self, days: list[date], max_delay: int) -> list[int]:
        """Per-ticket delays below ``max_delay`` for the given occurrence days."""
        out: list[int] = []
        for day in days:
            for delay, count in self.delay_counts.get(day, {}).items():
                if delay < max_delay:
                    out.extend([delay] * count)
        return out

    def delay_histogram(self, days: list[date], max_delay: int) -> list[int]:
        """Counts of delays ``0..max_delay-1`` for the given occurrence days."""
        hist = [0] * max_delay
        for day in days:
            for delay, count in self.delay_counts.get(day, {}).items():
                if 0 <= delay < max_delay:
                    hist[delay] += count
        return hist


@dataclass
class Dataset:
    rows: list[Row]
    cutoff: date
    channels: dict[str, ChannelData]

    @property
    def start(self) -> date:
        return min(row.occurrence for row in self.rows)

    @property
    def report_start(self) -> date:
        return min(row.report for row in self.rows)


def _parse_date(value: str, line: int, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError) as exc:
        raise DataError(f"line {line}: {field} is not an ISO date: {value!r}") from exc


def load_dataset(path: str | Path, cutoff: date | None = None) -> Dataset:
    """Read and validate the aggregate CSV.

    ``cutoff`` defaults to the latest report date present in the file.  Rows
    reported after an explicit cutoff are dropped, so the same loader can be
    used for replays without leaking future reports.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"data file not found: {path}")
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DataError("data file has no header row")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise DataError(f"data file is missing columns: {', '.join(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            channel = (raw["channel"] or "").strip()
            if not channel:
                raise DataError(f"line {line_no}: empty channel")
            occurrence = _parse_date(raw["occurrence_date"], line_no, "occurrence_date")
            report = _parse_date(raw["report_date"], line_no, "report_date")
            try:
                count = int(str(raw["count"]).strip())
            except (TypeError, ValueError) as exc:
                raise DataError(f"line {line_no}: count is not an integer: {raw['count']!r}") from exc
            if count < 0:
                raise DataError(f"line {line_no}: negative count {count}")
            if count == 0:
                continue
            if report < occurrence:
                raise DataError(f"line {line_no}: report_date precedes occurrence_date")
            rows.append(Row(channel, occurrence, report, count))
    if not rows:
        raise DataError("data file contains no usable rows")
    data_cutoff = cutoff or max(row.report for row in rows)
    visible = [row for row in rows if row.report <= data_cutoff]
    if not visible:
        raise DataError(f"no reports at or before cutoff {data_cutoff.isoformat()}")
    visible = [row for row in visible if row.occurrence <= data_cutoff]
    channels: dict[str, ChannelData] = {}
    for name in sorted({row.channel for row in visible}):
        channel_rows = [row for row in visible if row.channel == name]
        delay_counts: dict[date, dict[int, int]] = {}
        report_counts: dict[date, int] = {}
        for row in channel_rows:
            per_day = delay_counts.setdefault(row.occurrence, {})
            per_day[row.delay] = per_day.get(row.delay, 0) + row.count
            report_counts[row.report] = report_counts.get(row.report, 0) + row.count
        first = min(row.occurrence for row in channel_rows)
        channels[name] = ChannelData(
            name=name,
            first_occurrence=first,
            cutoff=data_cutoff,
            rows=channel_rows,
            delay_counts=delay_counts,
            report_counts=report_counts,
        )
    return Dataset(rows=visible, cutoff=data_cutoff, channels=channels)


def weekday_median(values: list[int]) -> float:
    """Median of a list of counts (used for outage baselines)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def poisson_like(values: list[int]) -> float:
    """A tiny helper used by tests and diagnostics."""
    if not values:
        return 0.0
    return sum(values) / math.sqrt(max(1, len(values)))
