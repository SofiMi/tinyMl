#!/usr/bin/env python3
"""Verify real GPT-2 receives only the current line, without roles or history."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
questions = ["hello", "i a ms sofia", "  A baby dragon met a boy named Ramil. Together, they"]
with tempfile.TemporaryDirectory() as folder:
    trace = Path(folder) / "trace.jsonl"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/gpt2_small.py"), "chat", "--trace", str(trace),
         "--max-tokens", "24"],
        input="\n".join(questions) + "\n/context\n/stats\n/new\n/context\n/quit\n",
        text=True, capture_output=True, check=True, timeout=120,
    )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
assert records[0]["mode"] == "independent-completion", records[0]
assert [r["prompt"] for r in records[1:]] == questions, records
assert all(r["seed"] is None for r in records[1:]), "RNG must not reset per input"
assert all(r["output_ids"] for r in records[1:]), records
contexts = re.findall(r"\[context-begin\]\n(.*?)\n\[context-end\]", result.stdout, re.S)
assert contexts == [questions[-1], ""], contexts
assert "history=0 role_prompt=0" in result.stdout, result.stdout
print("PASS: exact inputs (including spaces), no preceding turns or Ember prompt, real generation")
print("PASS: /context inspects last input only; /new clears inspection; RNG continuous")
print(result.stdout)
