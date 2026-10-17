#!/usr/bin/env python3
"""Replay a real model loop. Exit 1 when identical replies ignore distinct questions."""
import argparse
import json
from pathlib import Path

import torch
from dragon_hf import COMPACT_PERSONA, load, reply, sha


def replay(model, tokenizer, prompts, history=True):
    messages = [{"role": "system", "content": COMPACT_PERSONA}]
    turns = []
    for question in prompts:
        if not history:
            messages = messages[:1]
        messages.append({"role": "user", "content": question})
        answer, stats = reply(model, tokenizer, messages, "cpu", 48, 256)
        turns.append({"user": question, "assistant": answer, **stats})
        messages.append({"role": "assistant", "content": answer})
    return turns


def loops(turns):
    for i in range(len(turns)-2):
        window = turns[i:i+3]
        if any(t["user"] == "1 + 1?" for t in window) and len({t["assistant"] for t in window}) == 1 and window[0]["assistant"]:
            return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("build/dragon/dialogue-v1-retained/best"))
    p.add_argument("--prompts", type=Path, default=Path("tests/dragon/sofia-loop-v1.json"))
    p.add_argument("--output", type=Path, default=Path("build/dragon/sofia-loop.json"))
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--fresh", action="store_true", help="Diagnostic only: reset history before each question")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, tokenizer = load(a.model, "cpu")
    prompts = json.loads(a.prompts.read_text())
    turns = replay(model, tokenizer, prompts, not a.fresh)
    report = {"model": str(a.model), "checkpoint_sha256": sha(a.model/"model.safetensors"),
              "system": COMPACT_PERSONA, "history": not a.fresh, "context": 256, "max_tokens": 48,
              "sampling": "greedy", "turns": turns, "loop_detected": loops(turns)}
    for t in turns:
        print(f"You: {t['user']}\nBot: {t['assistant']}", flush=True)
    if a.minimize and loops(turns):
        current = prompts[:]
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                candidate = current[:i]+current[i+1:]
                if len(candidate) < 3:
                    continue
                trial = replay(model, tokenizer, candidate, not a.fresh)
                if loops(trial):
                    current = candidate
                    changed = True
                    break
        report["minimal_prompts"] = current
        report["minimal_turns"] = replay(model, tokenizer, current, not a.fresh)
        print("Minimal:", json.dumps(report["minimal_turns"], ensure_ascii=False), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print("FAIL: identical answers to changing questions" if report["loop_detected"] else "PASS: this repetition pattern was not detected (not a quality verdict)")
    raise SystemExit(1 if report["loop_detected"] else 0)


if __name__ == "__main__":
    main()
