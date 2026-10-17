import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import generate_dialogues_timed as timed


class TimedGenerationTests(unittest.TestCase):
    def test_deadline_interrupts_blocked_request(self):
        def slow_request():
            time.sleep(2)
        start = time.monotonic()
        call = timed.bounded_request(slow_request, start + .05)
        with self.assertRaises(timed.TimeBudgetExpired):
            call()
        self.assertLess(time.monotonic()-start, 1)

    def test_expired_budget_does_not_start_another_request(self):
        with patch.object(timed.data, "request") as request:
            call = timed.bounded_request(request, 1, clock=lambda: 2)
            with self.assertRaises(timed.TimeBudgetExpired):
                call()
            request.assert_not_called()

    def test_partial_request_is_not_committed_and_run_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = SimpleNamespace(output=directory, hours=.05/3600, count=1, seed=42,
                                   model="test", url="http://localhost:11434", min_free_mb=64)

            def request(base, endpoint, payload=None):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test", "digest": "sha256:test"}]}
                time.sleep(2)

            with patch.object(timed.data, "request", request), contextlib.redirect_stdout(io.StringIO()):
                timed.run(args)
                timed.run(args)
            status = json.loads((directory / "run-status.json").read_text())
            self.assertEqual(status["state"], "time_budget_reached")
            self.assertEqual(timed.data.journal_rows(directory), [])
            self.assertEqual(json.loads((directory / "report.json").read_text())["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
