#!/usr/bin/env python3
"""Evaluate or chat with a local Hugging Face dragon candidate (no network)."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from dragon_teacher import PERSONA

SMOL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
COMPACT_PERSONA = (
    "You are Ember, a friendly little dragon. You live in a warm cave and enjoy "
    "apples and shiny pebbles. Answer in short, simple English. "
    "Do not invent facts about the user."
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha(handle)


def _sha(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def load(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def fit_messages(tokenizer, messages, limit, template_options=None):
    messages = list(messages)
    while True:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False,
            **(template_options or {}),
        )
        if len(ids) <= limit:
            return ids
        # Preserve system prompt and the newest user turn; remove complete pairs.
        if len(messages) <= 2:
            raise ValueError("System message and question exceed the context budget")
        del messages[1:3]


@torch.inference_mode()
def reply(model, tokenizer, messages, device, max_tokens=48, context=1024,
          generation_options=None, template_options=None):
    ids = fit_messages(tokenizer, messages, context - max_tokens, template_options)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    started = time.monotonic()
    options = {"do_sample": False, **(generation_options or {})}
    output = model.generate(
        inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_tokens,
        use_cache=True, pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **options,
    )[0, len(ids):].tolist()
    stopped = bool(output and output[-1] == tokenizer.eos_token_id)
    if stopped:
        output = output[:-1]
    answer = tokenizer.decode(output, skip_special_tokens=True).strip()
    return answer, dict(input_tokens=len(ids), output_tokens=len(output),
                        stop="eos" if stopped else "length", seconds=time.monotonic()-started)


def evaluate(args, model, tokenizer, persona):
    report = {
        "model_directory": str(args.model), "source_revision": args.source_revision,
        "checkpoint_sha256": sha(args.model / "model.safetensors"),
        "tokenizer_sha256": sha(args.model / "tokenizer.json"),
        "chat_template": tokenizer.chat_template,
        "source_config": json.loads((args.model / "config.json").read_text()),
        "runtime_config": model.config.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": args.device, "weights": "FP32 runtime, original checkpoint precision recorded in source_config",
        "persona": persona, "sampling": args.generation_options, "template_options": args.template_options, "seed": args.seed, "max_tokens": args.max_tokens,
        "context": args.context, "cases_source": str(args.cases), "cases_sha256": sha(args.cases),
        "note": "Keyword checks are smoke checks, not semantic accuracy. Model uses its own generated history.",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, prompts, keywords in json.loads(args.cases.read_text()):
        messages = [{"role": "system", "content": persona}]
        turns = []
        for question in prompts:
            messages.append({"role": "user", "content": question})
            answer, details = reply(model, tokenizer, messages, args.device, args.max_tokens, args.context, args.generation_options, args.template_options)
            messages.append({"role": "assistant", "content": answer})
            turns.append(dict(user=question, assistant=answer, **details))
            print(f"{name}: {question}\nDragon: {answer}", flush=True)
        report["cases"].append(dict(id=name, turns=turns, expected_keywords=keywords,
                                    keyword_check=all(k in answer.lower() for k in keywords) if keywords else None))
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["eval", "chat"])
    parser.add_argument("--model", type=Path, default=Path.home()/".cache/tinyml-smollm2-135m-instruct")
    parser.add_argument("--source-revision", help="Pinned upstream commit, recorded in the report")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--persona", choices=["full", "compact", "assistant"], default="full")
    parser.add_argument("--cases", type=Path, default=Path("tests/dragon/heldout-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("build/dragon/smollm2-eval.json"))
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.source_revision is None and args.model == Path.home()/".cache/tinyml-smollm2-135m-instruct":
        args.source_revision = SMOL_REVISION
    if not 0 < args.max_tokens < args.context:
        parser.error("Require 0 < max-tokens < context")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.top_k < 0 or args.repetition_penalty <= 0:
        parser.error("Invalid sampling settings")
    args.generation_options = {"do_sample": args.temperature > 0, "repetition_penalty": args.repetition_penalty}
    if args.temperature:
        args.generation_options.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    args.template_options = {"enable_thinking": False} if args.no_thinking else {}
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model, tokenizer = load(args.model, args.device)
    if args.context > model.config.max_position_embeddings:
        parser.error("context exceeds model context")
    persona = {"full": PERSONA, "compact": COMPACT_PERSONA,
               "assistant": "You are a helpful AI assistant named SmolLM, trained by Hugging Face"}[args.persona]
    if args.mode == "eval":
        evaluate(args, model, tokenizer, persona)
        return
    messages = [{"role": "system", "content": persona}]
    print("Local candidate on this computer. /new resets history; /quit exits.")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/quit":
            break
        if question == "/new":
            messages = messages[:1]
            continue
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        answer, _ = reply(model, tokenizer, messages, args.device, args.max_tokens, args.context, args.generation_options, args.template_options)
        messages.append({"role": "assistant", "content": answer})
        print("Ember:", answer)


if __name__ == "__main__":
    main()
