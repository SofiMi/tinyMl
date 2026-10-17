#!/usr/bin/env python3
"""Replay the user's regression and natural cases against the installed local teacher."""
import json
from pathlib import Path

from dragon_teacher import request
from dragon_hf import COMPACT_PERSONA


def main():
    base, name = "http://127.0.0.1:11434", "qwen3.5:9b"
    model = next(m for m in request(base, "/api/tags")["models"] if m["name"] == name)
    cases = [("sofia_loop", json.loads(Path("tests/dragon/sofia-loop-v1.json").read_text()), [])]
    cases += json.loads(Path("tests/dragon/compression-v1.json").read_text())
    report = {"model": model, "system": COMPACT_PERSONA, "think": False,
              "options": {"num_ctx": 2048, "num_predict": 48, "temperature": 0, "seed": 42},
              "note": "Teacher control on Ollama/MLX; different context and runtime from small HF candidates, not a speed benchmark.",
              "cases": []}
    output = Path("docs/evaluation/reconsider/teacher.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for identifier, prompts, _ in cases:
        messages = [{"role": "system", "content": COMPACT_PERSONA}]
        turns = []
        for q in prompts:
            messages.append({"role": "user", "content": q})
            raw = request(base, "/api/chat", {"model": name, "messages": messages,
                "stream": False, "think": False, "keep_alive": "10m", "options": report["options"]})
            a = raw["message"]["content"].strip()
            messages.append({"role": "assistant", "content": a})
            turns.append({"user": q, "assistant": a, "done_reason": raw.get("done_reason"),
                          "eval_count": raw.get("eval_count")})
            print(f"{identifier}: {q}\nTeacher: {a}", flush=True)
        report["cases"].append({"id": identifier, "turns": turns})
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
