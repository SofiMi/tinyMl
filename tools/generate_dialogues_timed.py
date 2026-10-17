#!/usr/bin/env python3
"""Run local Qwen corpus generation for a bounded duration (8 hours by default)."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import dragon_distill_data as data


class TimeBudgetExpired(Exception):
    pass


def bounded_request(request, deadline, clock=time.monotonic):
    """Interrupt only network calls, never a journal append/export on disk.

    SIGALRM/setitimer are available on macOS/Linux; call from the main thread.
    An interrupted conversation is not committed and will be retried on resume.
    """
    def invoke(*args, **kwargs):
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeBudgetExpired()

        def expired(signum, frame):
            raise TimeBudgetExpired()

        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            return request(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    return invoke


def run(args):
    data.local_url(args.url)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["version"] != data.VERSION or plan["seed"] != args.seed or len(plan["rows"]) != args.count:
                raise ValueError("Existing plan differs; use its count/seed or a new output directory")
        else:
            plan = {"version": data.VERSION, "seed": args.seed, "rows": data.make_plan(args.count, args.seed)}
            data.write_json(path, plan)

        started = time.monotonic()
        status = {"started_unix": time.time(), "budget_seconds": args.hours * 3600,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "state": "running", "note": "Each invocation has its own time budget; completed dialogues are resumed."}
        data.write_json(args.output / "run-status.json", status)
        original = data.request
        data.request = bounded_request(original, started + args.hours * 3600)
        outcome = "failed"
        try:
            print(f"Generating for at most {args.hours:g} hours. Output: {args.output}", flush=True)
            data.generate(SimpleNamespace(**vars(args), limit=0))
            outcome = "plan_completed"
        except TimeBudgetExpired:
            outcome = "time_budget_reached"
            print("Time budget reached. Finished dialogues saved; unfinished dialogue will be retried.", flush=True)
        except KeyboardInterrupt:
            outcome = "interrupted"
            print("Interrupted. Finished dialogues saved.", flush=True)
        finally:
            data.request = original
            status.update(state=outcome, elapsed_seconds=time.monotonic()-started, finished_unix=time.time())
            data.write_json(args.output / "run-status.json", status)
            # generate() also exports on exit; this covers expiry during /api/tags.
            report = data.export(args.output)
            print(f"Saved: {report['attempts']} attempts, {report['accepted']} candidates. State: {outcome}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--output", type=Path, default=Path("build/dragon/qwen-8h"))
    parser.add_argument("--count", type=int, default=20000, help="maximum attempted dialogues, not a quality target")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--min-free-mb", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.hours <= 168 or args.count <= 0 or args.min_free_mb < 64:
        parser.error("hours must be in (0, 168], count > 0, reserve >= 64 MB")
    run(args)


if __name__ == "__main__":
    main()
