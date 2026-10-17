#!/usr/bin/env python3
"""Train a local HF student on general chat with variable system instructions."""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from dragon_hf import load, sha, fit_messages, reply


def examples(rows, tokenizer, context):
    result, skipped = [], 0
    for row in rows:
        messages = row["messages"]
        assert messages[0]["role"] == "system"
        for end in range(2, len(messages), 2):
            assert messages[end-1]["role"] == "user" and messages[end]["role"] == "assistant"
            history = messages[:end]
            prefix = tokenizer.apply_chat_template(history, tokenize=True, return_dict=False, add_generation_prompt=True)
            whole = tokenizer.apply_chat_template(messages[:end+1], tokenize=True, return_dict=False, add_generation_prompt=False)
            if whole[:len(prefix)] != prefix:
                raise ValueError("Chat template is not prefix-stable")
            target = whole[len(prefix):]
            if tokenizer.eos_token_id not in target:
                raise ValueError("Assistant target missing EOS")
            target = target[:target.index(tokenizer.eos_token_id)+1]
            try:
                prefix = fit_messages(tokenizer, history, context-len(target))
            except ValueError:
                skipped += 1
                continue
            result.append({"ids": prefix+target, "labels": [-100]*len(prefix)+target,
                           "family": row["family"], "row_id": row["id"]})
    return result, skipped


def batch(samples, pad, device):
    n = max(len(s["ids"]) for s in samples)
    ids = torch.full((len(samples), n), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, s in enumerate(samples):
        length = len(s["ids"])
        ids[i, :length] = torch.tensor(s["ids"])
        labels[i, :length] = torch.tensor(s["labels"])
        attention[i, :length] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": attention.to(device)}


def reference_kl(student_logits, reference_logits, labels, general_rows):
    """Forward KL(reference || student), on general assistant target positions."""
    mask = (labels[:, 1:] != -100) & general_rows[:, None]
    if not bool(mask.any()):
        return student_logits.sum()*0
    student = F.log_softmax(student_logits[:, :-1][mask], dim=-1)
    reference = F.log_softmax(reference_logits[:, :-1][mask].detach(), dim=-1)
    return F.kl_div(student, reference, log_target=True, reduction="batchmean")


@torch.inference_mode()
def validate(model, samples, tokenizer, device, size):
    model.eval()
    groups = {}
    for family in sorted({s["family"] for s in samples}):
        items = [s for s in samples if s["family"] == family]
        numerator, denominator = 0.0, 0
        for start in range(0, len(items), size):
            b = batch(items[start:start+size], tokenizer.pad_token_id, device)
            count = int((b["labels"][:, 1:] != -100).sum().item())
            loss = model(**b, use_cache=False).loss.item()
            numerator += loss*count
            denominator += count
        groups[family] = {"loss": numerator/denominator, "tokens": denominator}
    total = sum(g["loss"]*g["tokens"] for g in groups.values())/sum(g["tokens"] for g in groups.values())
    return {"loss": total, "families": groups}


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model, tokenizer = load(args.model, args.device)
    reference = None
    if args.reference_kl:
        reference, _ = load(args.model, args.device)
        reference.requires_grad_(False)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use an environment with GPU access or --device cpu")
    train_path, val_path = args.data/"train.jsonl", args.data/"validation.jsonl"
    rows = [json.loads(s) for s in train_path.read_text().splitlines()]
    val_rows = [json.loads(s) for s in val_path.read_text().splitlines()]
    assert not ({r["id"] for r in rows} & {r["id"] for r in val_rows})
    samples, skipped = examples(rows, tokenizer, args.context)
    valid, val_skipped = examples(val_rows, tokenizer, args.context)
    if not samples or not valid:
        raise ValueError("Empty train/validation after tokenization")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"training.json").exists():
        raise ValueError("Output already contains a run; select a new directory")
    initial = validate(model, valid, tokenizer, args.device, args.batch_size)
    report = {"base_model": str(args.model), "base_sha256": sha(args.model/"model.safetensors"),
              "train_sha256": sha(train_path), "validation_sha256": sha(val_path),
              "tokenizer_sha256": sha(args.model/"tokenizer.json"),
              "seed": args.seed, "device": args.device, "torch": torch.__version__,
              "parameters": sum(p.numel() for p in model.parameters()),
              "learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
              "context": args.context, "optimizer": "AdamW, weight_decay=0.01, clip=1, cosine to 10%",
              "objective": "assistant-only cross entropy including EOS; no persona-specific fixed identity",
              "reference_kl": args.reference_kl,
              "reference_kl_scope": "KL(base || student), temperature 1, general-family assistant target positions only",
              "general_loss_tolerance": args.general_loss_tolerance,
              "answers": len(samples), "validation_answers": len(valid),
              "family_answers": dict(Counter(s["family"] for s in samples)),
              "supervised_tokens_per_epoch": sum(sum(t != -100 for t in s["labels"][1:]) for s in samples),
              "skipped": skipped, "validation_skipped": val_skipped,
              "initial_validation": initial, "history": [], "best_epoch": 0}
    print(json.dumps(report), flush=True)
    (args.output/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = random.Random(args.seed)
    steps = math.ceil(len(samples)/args.batch_size)*args.epochs
    step, best = 0, initial["loss"]
    started = time.monotonic()
    for epoch in range(args.epochs):
        rng.shuffle(samples)
        model.train()
        numerator, denominator, kl_sum = 0., 0, 0.
        for start in range(0, len(samples), args.batch_size):
            step += 1
            lr = args.lr*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            items = samples[start:start+args.batch_size]
            b = batch(items, tokenizer.pad_token_id, args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**b, use_cache=False)
            ce = output.loss
            loss = ce
            if reference is not None and any(s["family"] == "general" for s in items):
                with torch.no_grad():
                    ref = reference(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits
                general = torch.tensor([s["family"] == "general" for s in items], device=args.device)
                kl = reference_kl(output.logits, ref, b["labels"], general)
                loss = ce + args.reference_kl*kl
                kl_sum += kl.item()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((b["labels"][:, 1:] != -100).sum().item())
            numerator += ce.item()*count
            denominator += count
            if step % 50 == 0:
                print(f"epoch={epoch+1} step={step}/{steps} loss={loss.item():.4f} seconds={time.monotonic()-started:.1f}", flush=True)
        val = validate(model, valid, tokenizer, args.device, args.batch_size)
        entry = {"epoch": epoch+1, "training_loss": numerator/denominator,
                 "reference_kl_mean_per_batch": kl_sum/math.ceil(len(samples)/args.batch_size),
                 "validation": val, "seconds": time.monotonic()-started}
        report["history"].append(entry)
        eligible = (args.general_loss_tolerance is None or
                    val["families"]["general"]["loss"] <= initial["families"]["general"]["loss"]*(1+args.general_loss_tolerance))
        entry["eligible_for_checkpoint"] = eligible
        if val["loss"] < best and eligible:
            best = val["loss"]
            report["best_epoch"] = epoch+1
            # Save CPU state without moving the training model or optimizer.
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            model.save_pretrained(args.output/"best", state_dict=state, safe_serialization=True)
            tokenizer.save_pretrained(args.output/"best")
        print(json.dumps(entry), flush=True)
        (args.output/"training.json").write_text(json.dumps(report, indent=2)+"\n")


def evaluate(args):
    torch.set_num_threads(4)
    model, tokenizer = load(args.model, args.device)
    report = {"model": str(args.model), "checkpoint_sha256": sha(args.model/"model.safetensors"),
              "cases_sha256": sha(args.cases), "context": args.context, "device": args.device,
              "max_tokens": 48, "sampling": "greedy", "cases": [],
              "note": "Strict final-answer format checks are narrow diagnostics, not general dialogue accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c in json.loads(args.cases.read_text()):
        messages = [{"role": "system", "content": c["system"]}]
        turns = []
        for q in c["prompts"]:
            messages.append({"role": "user", "content": q})
            answer, stats = reply(model, tokenizer, messages, args.device, 48, args.context)
            turns.append({"user": q, "assistant": answer, **stats})
            messages.append({"role": "assistant", "content": answer})
        normalized = answer.lower().strip().rstrip(".!?")
        passed = normalized in [s.lower() for s in c["answers"]] if c.get("answers") else None
        print(f"{c['id']} pass={passed}: {answer}", flush=True)
        report["cases"].append({**c, "turns": turns, "strict_check": passed})
        args.output.write_text(json.dumps(report, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("training/dialogue-v1"))
    p.add_argument("--cases", type=Path, default=Path("tests/dragon/dialogue-v1.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reference-kl", type=float, default=0., help="General-data KL penalty against frozen initial model")
    p.add_argument("--general-loss-tolerance", type=float, help="Maximum relative general validation loss increase for checkpoint selection")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.context <= 48 or args.reference_kl < 0 or (args.general_loss_tolerance is not None and args.general_loss_tolerance < 0):
        p.error("Invalid training/context parameters")
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
