#!/usr/bin/env python3
"""Fine-tune and evaluate stories15M locally, without changing ESP firmware."""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import time
import os
from pathlib import Path

import sentencepiece as spm
import torch
from dragon_teacher import CASES

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q4_roundtrip(weight):
    """Simulate the existing exporter's Q4_0 weights; arithmetic remains FP32."""
    blocks = weight.detach().cpu().float().reshape(-1, 32)
    largest = blocks.gather(1, blocks.abs().argmax(dim=1, keepdim=True))
    scale = largest / -8.0
    inverse = torch.where(scale != 0, 1.0 / scale, 0.0)
    codes = torch.clamp(torch.floor(blocks * inverse + 8.5), 0, 15)
    stored_scale = scale.to(torch.bfloat16).float()
    return ((codes - 8) * stored_scale).reshape(weight.shape)


def load(args):
    if args.architecture == "gpt-neo":
        from dragon_neo import NeoStudent, NeoTokenizer, GPTNeoConfig, initial_model
        tokenizer = NeoTokenizer(args.upstream)
        if args.checkpoint.suffix == ".safetensors":
            model, model_args = initial_model(args.upstream, args.checkpoint)
        else:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model_args = ckpt["model_args"]
            model = NeoStudent(GPTNeoConfig.from_dict(model_args["config"]))
            model.load_state_dict(ckpt["model"], strict=True)
        if args.fake_q4:
            raise ValueError("GPT-Neo needs its actual pruned export; Q4 simulation here is Llama-only")
    else:
        spec = importlib.util.spec_from_file_location("llama2c_model", args.upstream / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model_args = ckpt["model_args"]
        model = module.Transformer(module.ModelArgs(**model_args))
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state, strict=True)
        tokenizer = spm.SentencePieceProcessor(model_file=str(args.upstream / "tokenizer.model"))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    if args.fake_q4:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    parameter.copy_(q4_roundtrip(parameter).to(device))
    return model, tokenizer, model_args, device


def tokenizer_sha(args):
    return sha(args.upstream / ("tokenizer.json" if args.architecture == "gpt-neo" else "tokenizer.model"))


def prompt_tokens(sp, history, question):
    out = [sp.bos_id()]
    role = getattr(sp, "bot_role", "Dragon")
    for user, bot in history:
        out += sp.encode(f"User: {user}\n{role}:") + sp.encode(" " + bot) + [sp.eos_id()]
        out += sp.encode("\n")
    return out + sp.encode(f"User: {question}\n{role}:")


def fit_prompt(sp, history, question, limit):
    history = list(history)
    while True:
        ids = prompt_tokens(sp, history, question)
        if len(ids) <= limit:
            return ids
        if not history:
            raise ValueError("Question too long for the model context")
        history.pop(0)


@torch.inference_mode()
def generate(model, sp, history, question, device, max_tokens=48):
    model.eval()
    ids = fit_prompt(sp, history, question, model.params.max_seq_len - max_tokens)
    result = []
    stop = "length"
    for _ in range(max_tokens):
        x = torch.tensor([ids + result], dtype=torch.long, device=device)
        token = int(model(x)[0, -1].argmax().item())
        if token == sp.eos_id():
            stop = "eos"
            break
        result.append(token)
    return sp.decode(result).strip(), {"input_tokens": len(ids), "output_tokens": len(result), "stop": stop}


def evaluate(args, model, sp, device):
    report = {"checkpoint_sha256": sha(args.checkpoint), "tokenizer_sha256": tokenizer_sha(args),
              "architecture": args.architecture, "bot_role": getattr(sp, "bot_role", "Dragon"),
              "device": device, "sampling": "greedy", "max_tokens": 48,
              "weights": "simulated Q4_0 with BF16 scales, FP32 arithmetic" if args.fake_q4 else "FP32",
              "parameters": sum(p.numel() for p in model.parameters()), "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases is None else json.loads(args.cases.read_text())
    report["cases_source"] = "dragon_teacher.CASES" if args.cases is None else str(args.cases)
    for name, prompts, keywords in cases:
        history, turns = [], []
        for prompt in prompts:
            start = time.monotonic()
            answer, details = generate(model, sp, history, prompt, device)
            turns.append(dict(user=prompt, assistant=answer, seconds=time.monotonic()-start, **details))
            history.append((prompt, answer))
            print(f"{name}: {prompt}\nDragon: {answer}", flush=True)
        report["cases"].append({"id": name, "turns": turns,
                                "expected_keywords": keywords,
                                "keyword_check": all(k in answer.lower() for k in keywords) if keywords else None})
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def examples(path, sp, context):
    result = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        history = []
        messages = row["messages"]
        assert len(messages) % 2 == 0
        for i in range(0, len(messages), 2):
            user, bot = messages[i:i+2]
            assert user["role"] == "user" and bot["role"] == "assistant"
            target = sp.encode(" " + bot["content"]) + [sp.eos_id()]
            prefix = fit_prompt(sp, history, user["content"], context + 1 - len(target))
            tokens = prefix + target
            labels = [-1] * len(prefix) + target
            result.append((tokens[:-1], labels[1:]))
            history.append((user["content"], bot["content"]))
    return result


def batch(samples, device):
    width = max(len(x) for x, _ in samples)
    x = torch.zeros((len(samples), width), dtype=torch.long)
    y = torch.full((len(samples), width), -1, dtype=torch.long)
    for i, (a, b) in enumerate(samples):
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)
    return x.to(device), y.to(device)


@torch.inference_mode()
def validation_loss(model, samples, device, size):
    model.eval()
    numerator, denominator = 0.0, 0
    for start in range(0, len(samples), size):
        x, y = batch(samples[start:start+size], device)
        model(x, y)
        count = int((y != -1).sum().item())
        numerator += model.last_loss.item() * count
        denominator += count
    return numerator / denominator


def train(args, model, sp, model_args, device):
    if args.freeze_embeddings:
        model.tok_embeddings.weight.requires_grad_(False)
    train_path, val_path = args.data / "train.jsonl", args.data / "validation.jsonl"
    train_rows = [json.loads(s) for s in train_path.read_text().splitlines()]
    val_rows = [json.loads(s) for s in val_path.read_text().splitlines()]
    assert train_rows and val_rows
    assert not ({r["id"] for r in train_rows} & {r["id"] for r in val_rows})
    data = examples(train_path, sp, model.params.max_seq_len)
    valid = examples(val_path, sp, model.params.max_seq_len)
    print(f"device={device}, parameters={sum(p.numel() for p in model.parameters())}, train_answers={len(data)}, validation_answers={len(valid)}", flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source_checkpoint_sha256": sha(args.checkpoint), "tokenizer_sha256": tokenizer_sha(args),
              "architecture": args.architecture, "bot_role": getattr(sp, "bot_role", "Dragon"),
              "train_sha256": sha(train_path), "validation_sha256": sha(val_path),
              "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "device": device, "seed": 42, "lr": args.lr,
              "batch_size": args.batch_size, "epochs": args.epochs,
              "freeze_embeddings": args.freeze_embeddings,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "train_answers": len(data), "validation_answers": len(valid),
              "initial_validation_loss": validation_loss(model, valid, device, args.batch_size), "history": []}
    best = report["initial_validation_loss"]
    step = 0
    total_steps = args.epochs * math.ceil(len(data) / args.batch_size)
    rng = random.Random(42)
    start = time.monotonic()
    for epoch in range(args.epochs):
        rng.shuffle(data)
        model.train()
        total_loss = 0.0
        for offset in range(0, len(data), args.batch_size):
            step += 1
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = batch(data[offset:offset+args.batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            model(x, y)
            loss = model.last_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch={epoch+1} step={step}/{total_steps} loss={loss.item():.4f}", flush=True)
        val = validation_loss(model, valid, device, args.batch_size)
        entry = dict(epoch=epoch+1, training_loss=total_loss/math.ceil(len(data)/args.batch_size), validation_loss=val, seconds=time.monotonic()-start)
        report["history"].append(entry)
        print(json.dumps(entry), flush=True)
        if val < best:
            best = val
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "model_args": model_args, "epoch": epoch+1}, args.output / "best.pt")
            report["best_epoch"] = epoch + 1
        (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["eval", "train", "chat"])
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--output", type=Path, default=Path("build/dragon/student-eval.json"))
    p.add_argument("--data", type=Path, default=Path("build/dragon/data"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--cases", type=Path, help="Optional JSON list of [id, prompts, expected_keywords]")
    p.add_argument("--freeze-embeddings", action="store_true", help="Keep the tied input/output word table fixed during training")
    p.add_argument("--fake-q4", action="store_true", help="Simulate Q4 weights for evaluation/chat; not a firmware export")
    p.add_argument("--architecture", choices=["llama", "gpt-neo"], default="llama")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("epochs, batch-size and lr must be positive")
    if args.mode == "train" and args.fake_q4:
        p.error("--fake-q4 is only for eval/chat")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model, sp, model_args, device = load(args)
    if args.mode == "eval":
        evaluate(args, model, sp, device)
    elif args.mode == "train":
        train(args, model, sp, model_args, device)
    else:
        history = []
        print(f"Student {args.architecture}, greedy decoding. /new, /quit")
        while True:
            try:
                question = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question == "/quit":
                break
            if question == "/new":
                history = []
                continue
            if not question:
                continue
            try:
                answer, _ = generate(model, sp, history, question, device)
            except ValueError as error:
                print(error)
                continue
            print("Dragon:", answer)
            history.append((question, answer))


if __name__ == "__main__":
    main()
