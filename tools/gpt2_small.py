#!/usr/bin/env python3
"""Separate, pinned ComCom GPT-2 experiment on a computer (not ESP firmware)."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ComCom/gpt2-small"
REVISION = "2e765e0945e45ca54ac8b328737e7ffc72f2bb59"
WEIGHTS_SHA256 = "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421"
FILES = ("config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md")
PREAMBLE = ("This is a conversation with Ember, a friendly little dragon. "
            "Ember answers in short English sentences.\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise RuntimeError("Unexpected upstream revision")
    entries = {item["rfilename"]: item for item in metadata["siblings"]}
    needed = sum(entries[name]["size"] for name in FILES if not (folder / name).exists())
    if needed and shutil.disk_usage(folder).free < needed + 1024 ** 3:
        raise RuntimeError("Need room for download plus 1 GiB free reserve")
    checksums = {}
    for name in FILES:
        path = folder / name
        entry = entries[name]
        if not path.exists():
            print(f"Downloading {name} ({entry['size']:,} bytes)", flush=True)
            partial = path.with_name(name + ".partial")
            with urllib.request.urlopen(
                f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120
            ) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(path)
        if path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Size mismatch: {path}; remove only this incomplete file and retry")
        sha = digest(path)
        if name == "pytorch_model.bin":
            if sha != WEIGHTS_SHA256 or sha != entry["lfs"]["sha256"]:
                raise RuntimeError("Weights checksum mismatch")
        else:
            blob = path.read_bytes()
            git_sha = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
            if git_sha != entry["blobId"]:
                raise RuntimeError(f"Upstream blob mismatch: {name}")
        checksums[name] = {"bytes": path.stat().st_size, "sha256": sha}
    manifest = {"model": MODEL, "revision": REVISION, "files": checksums}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified model: {folder}")


class Runner:
    def __init__(self, folder, threads):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(threads)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["model"] != MODEL or manifest["revision"] != REVISION:
            raise RuntimeError("Wrong model manifest")
        for name in FILES:
            if digest(folder / name) != manifest["files"][name]["sha256"]:
                raise RuntimeError(f"Changed local model file: {name}")
        if manifest["files"]["pytorch_model.bin"]["sha256"] != WEIGHTS_SHA256:
            raise RuntimeError("Wrong model weights")
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            weights_only=True, attn_implementation="eager", dtype=torch.float32,
        ).eval()
        self.info = dict(manifest, parameters=sum(p.numel() for p in self.model.parameters()),
                         platform=platform.platform(), torch=torch.__version__,
                         transformers=transformers.__version__, device="cpu", dtype="float32",
                         threads=threads, context=self.model.config.n_positions)

    def generate(self, prompt, temperature, seed, max_tokens):
        torch = self.torch
        # A conversation seeds once; resetting here on every turn can lock the
        # base model into exactly the same answer despite changing inputs.
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = inputs.input_ids.shape[1]
        if count + max_tokens > self.model.config.n_positions:
            raise ValueError(f"Input ({count}) + reply ({max_tokens}) exceeds "
                             f"{self.model.config.n_positions} tokens; shorten the text")
        options = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                       pad_token_id=self.tokenizer.eos_token_id,
                       eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.9, top_k=50)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        elapsed = time.perf_counter() - started
        ids = output[0, count:].tolist()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        return dict(prompt=prompt, raw_completion=raw, output_ids=ids,
                    input_tokens=count, generated_tokens=len(ids), seconds=elapsed,
                    temperature=temperature, seed=seed, max_new_tokens=max_tokens,
                    sampling_top_p=0.9 if temperature > 0 else None,
                    sampling_top_k=50 if temperature > 0 else None)


def reply_text(raw):
    # GPT-2 has no chat template. Keep the full completion in evaluation records.
    end = min([raw.find(marker) for marker in ("\nUser:", "\nEmber:")
               if marker in raw] or [len(raw)])
    return raw[:end].strip()


def evaluate(runner, output, max_tokens):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidentally replacing an earlier experiment.
    with output.open("x") as f:
        report = {"provenance": runner.info, "cases": [], "dialogues": []}
        cases = [
            ("story", "Once upon a time, there was a little dragon named Ember."),
            ("fact_completion", "The capital of France is"),
            ("dragon_role", PREAMBLE + "User: Hello! Who are you?\nEmber:"),
            ("unknown_fact", PREAMBLE + "User: What is my favorite color?\nEmber:"),
        ]
        questions = ["Hello! My name is Sofia. What is your name?",
                     "What is my name?", "Actually, my name is Anna, not Sofia.",
                     "What color is grass?", "What is my name now?",
                     "What is your name, and what is my name?",
                     "What is my favorite food?"]
        for temperature in (0.0, 0.7):
            for name, prompt in cases:
                result = runner.generate(prompt, temperature, 42, max_tokens)
                report["cases"].append(dict(name=name, **result))
                print(f"{name} T={temperature}: {result['raw_completion']!r}", flush=True)
            runner.torch.manual_seed(42)
            history = PREAMBLE
            turns = []
            for question in questions:
                prompt = history + f"User: {question}\nEmber:"
                result = runner.generate(prompt, temperature, None, max_tokens)
                answer = reply_text(result["raw_completion"])
                turns.append(dict(user=question, displayed_answer=answer, **result))
                history = prompt + " " + answer + "\n"
                print(f"dialogue T={temperature}: {question} -> {answer!r}", flush=True)
            report["dialogues"].append(dict(temperature=temperature, session_seed=42, turns=turns))
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "eval", "chat", "complete"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "build/gpt2-small/model")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gpt2-small/report.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--trace", type=Path, help="Save exact interactive inputs/outputs as a new JSONL file")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.max_tokens <= 512 or args.temperature < 0:
        parser.error("threads >= 1, max-tokens in 1..512, temperature >= 0 required")
    if args.trace and args.mode != "chat":
        parser.error("--trace is for chat mode")
    if args.mode == "fetch":
        fetch(args.model_dir)
        return
    runner = Runner(args.model_dir, args.threads)
    print(f"{MODEL}: {runner.info['parameters']:,} parameters, CPU FP32", flush=True)
    if args.mode == "eval":
        evaluate(runner, args.output, args.max_tokens)
    elif args.mode == "complete":
        result = runner.generate(args.prompt, args.temperature, args.seed, args.max_tokens)
        print(args.prompt + result["raw_completion"])
    else:
        print("GPT-2 text completion: each input is independent; no history or role prompt.")
        print("/context shows the last input; /stats shows its size; /quit exits.")
        runner.torch.manual_seed(args.seed)
        last_input = ""
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("x") if args.trace else nullcontext() as trace:
            if trace:
                trace.write(json.dumps(dict(provenance=runner.info, session_seed=args.seed,
                                            mode="independent-completion")) + "\n")
            while True:
                try:
                    prompt = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                command = prompt.strip()
                if command == "/quit":
                    break
                if command == "/new":
                    last_input = ""
                    print("[every input already starts with empty context]")
                    continue
                if command == "/context":
                    print("[context-begin]\n" + last_input + "\n[context-end]")
                    continue
                if command == "/stats":
                    count = len(runner.tokenizer.encode(last_input, add_special_tokens=False))
                    print(f"[context] history=0 role_prompt=0 last_input_tokens={count} "
                          f"limit={runner.model.config.n_positions} reply_reserve={args.max_tokens}")
                    continue
                if not command:
                    continue
                try:
                    result = runner.generate(prompt, args.temperature, None, args.max_tokens)
                except ValueError as error:
                    print(error)
                    continue
                last_input = prompt  # inspection only; never passed to the next call
                print("GPT-2:", result["raw_completion"])
                if trace:
                    trace.write(json.dumps(result, ensure_ascii=False) + "\n")
                    trace.flush()


if __name__ == "__main__":
    main()
