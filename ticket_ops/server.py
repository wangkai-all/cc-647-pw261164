"""Small JSON API and static file server, standard library only.

    python -m ticket_ops.server --data data/tickets.csv --port 8000
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .data import DataError, Dataset, load_dataset
from .model import (
    DEFAULT_DRAWS,
    DEFAULT_SMOOTH,
    DEFAULT_WINDOW,
    SMOOTH_MAX,
    ChannelEstimate,
    DelayProfile,
    aggregate,
    estimate_channel,
    estimate_dataset,
    replay,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"
CACHE_LIMIT = 24


def _num(value: float) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), 3)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class EstimatorService:
    """Holds the dataset and an LRU cache of estimates keyed by parameters."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self._lock = threading.Lock()
        self._cache: dict[tuple, dict[str, ChannelEstimate]] = {}
        self._order: list[tuple] = []
        self.dataset: Dataset | None = None
        self.loaded_mtime: float | None = None

    def dataset_now(self) -> Dataset:
        mtime = self.data_path.stat().st_mtime
        if self.dataset is None or mtime != self.loaded_mtime:
            self.dataset = load_dataset(self.data_path)
            self.loaded_mtime = mtime
            self._cache.clear()
            self._order.clear()
        return self.dataset

    def estimates(self, window: int, smooth: float, draws: int) -> tuple[Dataset, dict[str, ChannelEstimate]]:
        ds = self.dataset_now()
        key = (window, round(smooth, 3), draws, ds.cutoff.isoformat(), len(ds.rows))
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return ds, self._cache[key]
        result = estimate_dataset(ds, window=window, smooth=smooth, draws=draws)
        with self._lock:
            self._cache[key] = result
            self._order.append(key)
            while len(self._order) > CACHE_LIMIT:
                stale = self._order.pop(0)
                self._cache.pop(stale, None)
        return ds, result


def _profile_payload(profile: DelayProfile | None) -> dict | None:
    if profile is None:
        return None
    return {
        "horizon": profile.horizon,
        "max_delay": profile.max_delay,
        "capped": profile.capped,
        "mean_delay": _num(profile.mean_delay),
        "train_days": profile.train_days,
        "train_tickets": profile.train_tickets,
        "bootstrap_tv95": _num(profile.bootstrap_tv95),
        "probs": [_num(x) for x in profile.probs.tolist()],
    }


def _channel_payload(est: ChannelEstimate) -> dict:
    days = [
        {
            "day": d.day.isoformat(),
            "age": d.age,
            "observed": d.observed,
            "estimated": _num(d.estimated),
            "q05": _num(d.q05),
            "q25": _num(d.q25),
            "q75": _num(d.q75),
            "q95": _num(d.q95),
            "unreported": _num(d.unreported),
            "observed_in_horizon": d.observed_in_horizon,
            "long_observed": d.long_observed,
            "matured": d.matured,
            "lower_bound": d.lower_bound,
            "contribution": [_num(x) for x in d.contribution],
            "observed_by_delay": d.observed_by_delay,
        }
        for d in est.days
    ]
    return {
        "channel": est.channel,
        "status": est.status,
        "cutoff": est.cutoff.isoformat(),
        "profile": _profile_payload(est.profile),
        "days": days,
        "observed_reports": est.observed_reports,
        "predicted_reports": {k: _num(v) for k, v in est.predicted_reports.items()},
        "rmse": _num(est.rmse),
        "warnings": est.warnings,
        "note": est.note,
    }


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def build_api(service: EstimatorService) -> dict:
    def estimates(query: dict[str, list[str]]) -> dict:
        window = int(_clamp(_to_int(_first(query, "window"), DEFAULT_WINDOW), 7, 60))
        smooth = _clamp(_to_float(_first(query, "smooth"), DEFAULT_SMOOTH), 0.0, SMOOTH_MAX)
        draws = int(_clamp(_to_int(_first(query, "draws"), DEFAULT_DRAWS), 0, 400))
        ds, ests = service.estimates(window, smooth, draws)
        channels = {name: _channel_payload(est) for name, est in ests.items()}
        return {
            "cutoff": ds.cutoff.isoformat(),
            "start": ds.start.isoformat(),
            "params": {"window": window, "smooth": smooth, "draws": draws, "smooth_max": SMOOTH_MAX},
            "channels": channels,
            "aggregate": aggregate(ests),
            "channel_order": list(ests.keys()),
        }

    def replays(query: dict[str, list[str]]) -> dict:
        window = int(_clamp(_to_int(_first(query, "window"), DEFAULT_WINDOW), 7, 60))
        smooth = _clamp(_to_float(_first(query, "smooth"), DEFAULT_SMOOTH), 0.0, SMOOTH_MAX)
        ds = service.dataset_now()
        offsets_raw = _first(query, "offsets")
        if offsets_raw:
            offsets = [int(x) for x in offsets_raw.split(",") if x.strip().lstrip("-").isdigit()]
        else:
            offsets = [28, 21, 14, 7]
        cutoffs = [ds.cutoff.fromordinal(ds.cutoff.toordinal() - o) for o in offsets if o > 0]
        result = replay(
            service.data_path,
            current_cutoff=ds.cutoff,
            replay_cutoffs=cutoffs,
            window=window,
            smooth=smooth,
            draws=40,
        )
        return result

    def meta(query: dict[str, list[str]]) -> dict:
        ds = service.dataset_now()
        channel_info = {}
        for name, ch in ds.channels.items():
            channel_info[name] = {
                "first_occurrence": ch.first_occurrence.isoformat(),
                "tickets": sum(ch.observed_total(d) for d in ch.occurrence_days),
                "last_report": max(ch.report_counts).isoformat() if ch.report_counts else None,
            }
        return {
            "cutoff": ds.cutoff.isoformat(),
            "start": ds.start.isoformat(),
            "channels": channel_info,
            "smooth_max": SMOOTH_MAX,
            "defaults": {"window": DEFAULT_WINDOW, "smooth": DEFAULT_SMOOTH},
        }

    return {"estimates": estimates, "replays": replays, "meta": meta}


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def make_handler(service: EstimatorService):
    api = build_api(service)

    class Handler(BaseHTTPRequestHandler):
        server_version = "TicketOps/1.0"

        def log_message(self, fmt, *args):  # quieter default logging
            print(f"{self.address_string()} - {fmt % args}")

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                name = parsed.path[len("/api/") :]
                fn = api.get(name)
                if fn is None:
                    self._send_json({"error": f"unknown endpoint {name}"}, status=404)
                    return
                query = parse_qs(parsed.query)
                try:
                    self._send_json(fn(query))
                except DataError as exc:
                    self._send_json({"error": str(exc)}, status=422)
                except Exception as exc:  # pragma: no cover - surfaces in the UI
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return
            self._serve_static(parsed.path)

        def _serve_static(self, path: str) -> None:
            rel = path.lstrip("/") or "index.html"
            if not rel.endswith((".html", ".css", ".js", ".svg", ".json", ".map", ".md")):
                rel = "index.html"
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR not in target.parents and target != STATIC_DIR:
                self.send_error(403)
                return
            if not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml",
                ".json": "application/json",
                ".map": "application/json",
                ".md": "text/markdown; charset=utf-8",
            }
            self.send_response(200)
            self.send_header("Content-Type", types.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(data_path: Path, host: str, port: int) -> None:
    service = EstimatorService(data_path)
    service.dataset_now()  # fail fast on a bad file
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"ticket-ops estimation page: http://{shown}:{port}/  (data: {data_path})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.data, args.host, args.port)


if __name__ == "__main__":
    main()
