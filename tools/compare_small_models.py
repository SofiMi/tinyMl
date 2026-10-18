#!/usr/bin/env python3
"""Small, inspectable Q4/Q8 comparison. Raw replies, no keyword quality score."""
import argparse
import json
from pathlib import Path
import time

from small_models import GGUFRunner, LOCK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cases = ["Hello! My name is Sofia. What is your name?",
             "What color is grass? Answer in one short sentence.",
             "Write a short story about a baby dragon and a boy named Ramil."]
    dialogue = ["Hello! My name is Sofia. What is your name?", "What is my name?",
                "Actually, my name is Anna, not Sofia.", "What color is grass?",
                "What is my name now?", "What is my favorite food?"]
    settings = argparse.Namespace(raw=False, threads=2, temperature=0.0, max_tokens=64)
    for key in ("supra", "supra-q8", "stentor", "stentor-q8"):
        runner = GGUFRunner(key, settings)
        with (a.output / f"{key}.jsonl").open("x") as output:
            history = []
            for group, prompts in [("independent", cases), ("dialogue", dialogue)]:
                settings.max_tokens = 64 if group == "independent" else 32
                for prompt in prompts:
                    started = time.monotonic()
                    result = runner.generate(prompt, history)
                    result.update(user=prompt, group=group, model=key, info=runner.info,
                                  revision=LOCK[key]["revision"], temperature=0.0,
                                  max_tokens=settings.max_tokens, seconds=time.monotonic()-started)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{key} / {group}: {prompt}\n{result['answer']}\n", flush=True)
                    if group == "dialogue":
                        history.extend([dict(role="user", content=prompt),
                                        dict(role="assistant", content=result["answer"])])
        runner.model.close()


if __name__ == "__main__":
    main()
