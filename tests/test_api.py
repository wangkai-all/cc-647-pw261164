"""Server-side smoke tests for the JSON API (no browser needed).

Runs the real stdlib server against the bundled dataset on an ephemeral port
and checks the payload contract the page relies on.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

from ticket_ops.server import EstimatorService, make_handler

DATA = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        service = EstimatorService(DATA)
        service.dataset_now()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=60) as res:
            return json.loads(res.read().decode("utf-8"))

    def test_meta(self) -> None:
        meta = self.get("/api/meta")
        self.assertEqual(meta["cutoff"], "2026-11-22")
        self.assertGreaterEqual(len(meta["channels"]), 5)

    def test_estimates_contract(self) -> None:
        payload = self.get("/api/estimates?window=28&smooth=0.3&draws=40")
        self.assertEqual(payload["params"]["window"], 28)
        for name, channel in payload["channels"].items():
            self.assertEqual(channel["status"], "ok", name)
            self.assertTrue(channel["days"], name)
            last = channel["days"][-1]
            self.assertFalse(last["matured"])
            self.assertGreaterEqual(last["estimated"], last["observed"])
            self.assertLessEqual(last["q05"], last["estimated"] + 1e-6)
            self.assertGreaterEqual(last["q95"], last["estimated"] - 1e-6)
            self.assertEqual(len(last["contribution"]), channel["profile"]["horizon"])
            self.assertEqual(len(last["observed_by_delay"]), channel["profile"]["horizon"])
            for value in (last["estimated"], last["q05"], last["q95"]):
                self.assertGreaterEqual(value, 0.0, name)
        # Aggregate must be a plain sum of per-channel point estimates.
        day = payload["aggregate"]["days"][-1]["day"]
        total = sum(
            next(d for d in ch["days"] if d["day"] == day)["estimated"]
            for ch in payload["channels"].values()
        )
        self.assertAlmostEqual(payload["aggregate"]["days"][-1]["estimated"], total, places=1)

    def test_predicted_reports_extend_past_cutoff(self) -> None:
        payload = self.get("/api/estimates?window=28&smooth=0.3&draws=20")
        for channel in payload["channels"].values():
            horizon = channel["profile"]["horizon"]
            last_pred = max(channel["predicted_reports"])
            self.assertGreater(last_pred, payload["cutoff"])
            self.assertLessEqual(
                date.fromisoformat(last_pred).toordinal() - date.fromisoformat(payload["cutoff"]).toordinal(),
                horizon,
            )

    def test_warnings_present(self) -> None:
        payload = self.get("/api/estimates?window=28&smooth=0.3&draws=20")
        kinds = {w["kind"]: name for name, ch in payload["channels"].items() for w in ch["warnings"]}
        self.assertIn("slowdown", kinds)
        self.assertIn("outage", kinds)
        self.assertIn("uncovered_delay", kinds)
        self.assertEqual(kinds["outage"], "电话")
        self.assertEqual(kinds["slowdown"], "邮件")

    def test_replays_use_only_past_data(self) -> None:
        payload = self.get("/api/replays?window=28&smooth=0.3")
        self.assertTrue(payload["replays"])
        today = date.fromisoformat(payload["current_cutoff"])
        for replay in payload["replays"]:
            cutoff = date.fromisoformat(replay["cutoff"])
            self.assertLess(cutoff, today)
            for channel in replay["channels"].values():
                for pair in channel["pairs"]:
                    if pair["immature_then"]:
                        # The value that "later arrived" can only be non-negative.
                        self.assertGreaterEqual(pair["later_arrived"], -1e-9)
                    self.assertGreaterEqual(pair["matured_now"], pair["observed_then"] - 1e-9)

    def test_smoothing_changes_estimates_but_not_constraints(self) -> None:
        raw = self.get("/api/estimates?window=28&smooth=0&draws=0")
        smooth = self.get("/api/estimates?window=28&smooth=8&draws=0")
        changed = 0
        for name, channel in raw["channels"].items():
            other = smooth["channels"][name]
            for a, b in zip(channel["days"], other["days"]):
                self.assertGreaterEqual(a["estimated"], a["observed"] - 1e-6)
                self.assertGreaterEqual(b["estimated"], b["observed"] - 1e-6)
                if abs(a["estimated"] - b["estimated"]) > 0.5:
                    changed += 1
        self.assertGreater(changed, 0)

    def test_bad_parameter_is_clamped(self) -> None:
        payload = self.get("/api/estimates?window=9999&smooth=-5&draws=1")
        self.assertEqual(payload["params"]["window"], 60)
        self.assertEqual(payload["params"]["smooth"], 0.0)


if __name__ == "__main__":
    unittest.main()
