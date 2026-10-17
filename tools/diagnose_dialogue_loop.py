#!/usr/bin/env python3
"""Replay a real model loop. Exit 1 when identical replies ignore distinct questions."""
import argparse
import json
from pathlib import Path

import torch
from dragon_hf import COMPACT_PERSONA, load, reply, sha


def replay(model, tokenizer, prompts, history=True, generation_options=None, template_options=None):
    messages = [{"role": "system", "content": COMPACT_PERSONA}]
    turns = []
    for question in prompts:
        if not history:
            messages = messages[:1]
        messages.append({"role": "user", "content": question})
        answer, stats = reply(model, tokenizer, messages, "cpu", 48, 256, generation_options, template_options)
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
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--top-p", type=float, default=.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-thinking", action="store_true")
    a = p.parse_args()
    if a.minimize and a.temperature:
        p.error("Minimization requires greedy generation")
    if not 0 <= a.temperature <= 2 or not 0 < a.top_p <= 1 or a.top_k < 0 or a.repetition_penalty <= 0:
        p.error("Invalid sampling settings")
    options = {"do_sample": a.temperature > 0, "repetition_penalty": a.repetition_penalty}
    if a.temperature:
        options.update(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    template = {"enable_thinking": False} if a.no_thinking else {}
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    model, tokenizer = load(a.model, "cpu")
    prompts = json.loads(a.prompts.read_text())
    turns = replay(model, tokenizer, prompts, not a.fresh, options, template)
    report = {"model": str(a.model), "checkpoint_sha256": sha(a.model/"model.safetensors"),
              "system": COMPACT_PERSONA, "history": not a.fresh, "context": 256, "max_tokens": 48,
              "sampling": options, "template_options": template, "seed": a.seed,
              "parameters": sum(p.numel() for p in model.parameters()),
              "turns": turns, "loop_detected": loops(turns)}
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
                trial = replay(model, tokenizer, candidate, not a.fresh, options, template)
                if loops(trial):
                    current = candidate
                    changed = True
                    break
        report["minimal_prompts"] = current
        report["minimal_turns"] = replay(model, tokenizer, current, not a.fresh, options, template)
        print("Minimal:", json.dumps(report["minimal_turns"], ensure_ascii=False), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print("FAIL: identical answers to changing questions" if report["loop_detected"] else "PASS: this repetition pattern was not detected (not a quality verdict)")
    raise SystemExit(1 if report["loop_detected"] else 0)


if __name__ == "__main__":
    main()
