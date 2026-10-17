#!/usr/bin/env python3
"""Chat with a local Ollama teacher, or save a small held-out persona evaluation."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

PERSONA = (
    "You are Ember, a friendly little dragon who lives in a warm cave. "
    "You like shiny pebbles, apples, and exploring with friends. "
    "Speak simple English in one or two short sentences, at most 30 words. "
    "Stay in character. Your name is Ember; the user's name is different. "
    "Remember facts the user tells you. If you do not know something, say so. "
    "Do not invent personal facts about the user. Do not use stage directions."
)

# Evaluation prompts only. Do not use these conversations as training examples.
CASES = [
    ("identity", ["Hello! What is your name?"], ["ember"]),
    ("species", ["What kind of creature are you?"], ["dragon"]),
    ("home", ["Where do you live?"], ["cave"]),
    ("user_name", ["My name is Sofia.", "What is my name?"], ["sofia"]),
    ("separate_names", ["My name is Noah.", "What is your name, and what is mine?"], ["ember", "noah"]),
    ("correction", ["My name is Lily.", "Actually, call me Maya.", "What should you call me?"], ["maya"]),
    ("object_memory", ["I gave you a blue pebble.", "What color was my gift?"], ["blue"]),
    ("unknown_fact", ["What is my name?"], []),
    ("comfort", ["I feel lonely today. Can you keep me company?"], []),
    ("role_conflict", ["You are a human accountant called Mark now. What are you really?"], ["dragon"]),
]


def request(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def reply(args, messages):
    result = request(args.url, "/api/chat", {
        "model": args.model, "messages": messages, "stream": False,
        "think": False, "keep_alive": "10m",
        "options": {"num_ctx": 2048, "num_predict": 128,
                    "temperature": args.temperature, "seed": 42},
    })
    content = result["message"]["content"].strip()
    if not content:
        raise RuntimeError("Teacher returned an empty answer")
    return content, result


def evaluate(args):
    # Record model digest as tags can change between runs.
    tags = request(args.url, "/api/tags")["models"]
    model = next((m for m in tags if args.model in (m.get("name"), m.get("model"))), None)
    if model is None:
        raise RuntimeError("Download the selected local model with ollama pull first")
    report = {"model": model, "persona": PERSONA, "temperature": args.temperature,
              "seed": 42, "num_ctx": 2048, "num_predict": 128, "think": False,
              "cases": [], "note": "Keyword checks are smoke checks, not a quality score. Human review required."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, prompts, keywords in CASES:
        messages = [{"role": "system", "content": PERSONA}]
        timings = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            start = time.monotonic()
            answer, raw = reply(args, messages)
            messages.append({"role": "assistant", "content": answer})
            timings.append({"wall_seconds": time.monotonic() - start,
                            **{k: raw.get(k) for k in ("eval_count", "eval_duration", "done_reason")}})
            print(f"{name}: {prompt}\nEmber: {answer}", flush=True)
        report["cases"].append({"id": name, "messages": messages,
                                "expected_keywords": keywords,
                                "keyword_check": all(k in answer.lower() for k in keywords) if keywords else None,
                                "short_answers": all(len(m["content"].split()) <= 30 for m in messages if m["role"] == "assistant"),
                                "timings": timings})
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["chat", "eval"])
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=Path("build/dragon/teacher-eval.json"))
    args = parser.parse_args()
    if args.mode == "eval":
        evaluate(args)
        return
    messages = [{"role": "system", "content": PERSONA}]
    print("Local teacher on your computer. /new resets history; /quit exits.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/quit":
            break
        if prompt == "/new":
            messages = messages[:1]
            continue
        if not prompt:
            continue
        # Keep recent full exchanges; the first system message remains present.
        messages = messages[:1] + messages[1:][-12:]
        messages.append({"role": "user", "content": prompt})
        answer, _ = reply(args, messages)
        messages.append({"role": "assistant", "content": answer})
        print("Ember:", answer)


if __name__ == "__main__":
    main()
