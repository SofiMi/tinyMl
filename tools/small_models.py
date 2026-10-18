#!/usr/bin/env python3
"""Pinned local playground: Stories42M Q4, Stentor30M Q4, VaghLM checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "build/model-playground"
LOCK = json.loads(Path(__file__).with_name("small_models.lock.json").read_text())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if "lfs" in entry:
        return sha256(path) == entry["lfs"]["sha256"]
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["blobId"]


def fetch(key):
    spec = LOCK[key]
    folder = STORE / key
    folder.mkdir(parents=True, exist_ok=True)
    needed = sum(e["size"] for e in spec["files"] if not verify(folder / e["rfilename"], e))
    if shutil.disk_usage(folder).free < needed + 1024**3:
        raise RuntimeError("Not enough disk space: leave at least 1 GiB free")
    manifest = dict(repo=spec["repo"], revision=spec["revision"], files={})
    for e in spec["files"]:
        path = folder / e["rfilename"]
        if not verify(path, e):
            if path.exists():
                raise RuntimeError(f"Changed/incomplete file: {path}; move it aside before retrying")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            print(f"Downloading {key}/{e['rfilename']} ({e['size']:,} bytes)", flush=True)
            repo, rev = e.get("repo", spec["repo"]), e.get("revision", spec["revision"])
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{e['rfilename']}"
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if not verify(partial, e):
                raise RuntimeError(f"Checksum mismatch: {partial}")
            partial.replace(path)
        manifest["files"][e["rfilename"]] = dict(bytes=e["size"], sha256=sha256(path))
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {key}: {folder}", flush=True)


def verify_local(key):
    for entry in LOCK[key]["files"]:
        if not verify(STORE / key / entry["rfilename"], entry):
            raise RuntimeError(f"Missing/changed {entry['rfilename']}; run fetch {key}")


class GGUFRunner:
    def __init__(self, key, args):
        self.key, self.args = key, args
        verify_local(key)
        import llama_cpp
        from tokenizers import Tokenizer
        filename = next(e["rfilename"] for e in LOCK[key]["files"] if e["rfilename"].endswith(".gguf"))
        self.tokenizer = Tokenizer.from_file(str(STORE / key / "tokenizer.json"))
        from llama_cpp._utils import suppress_stdout_stderr
        with suppress_stdout_stderr(disable=False):
            self.model = llama_cpp.Llama(model_path=str(STORE / key / filename),
                                         n_ctx=512 if key == "stentor" else 1024,
                                         n_threads=args.threads, n_gpu_layers=0, seed=42, verbose=False)
        if self.model.n_vocab() != self.tokenizer.get_vocab_size():
            raise RuntimeError("HF/GGUF vocabulary sizes differ")
        self.template = None
        if key == "stentor":
            from jinja2 import StrictUndefined
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(undefined=StrictUndefined)
            def fail(message):
                raise ValueError(message)
            env.globals["raise_exception"] = fail
            self.template = env.from_string((STORE / key / "chat_template.jinja").read_text())
        self.info = dict(backend="llama-cpp-python", version=llama_cpp.__version__,
                         quantization="Q4_K_M", device="cpu", context=self.model.n_ctx(),
                         parameters=30419712 if key == "stentor" else 58073600,
                         tokenizer="original Hugging Face tokenizer.json; direct token IDs",
                         weights_sha256=sha256(STORE / key / filename))
        # Compare every token ID, not just vocab sizes, before bypassing GGUF tokenization.
        for token, i in self.tokenizer.get_vocab().items():
            actual = self.model._model.token_get_text(i)
            if actual != token:
                raise RuntimeError(f"HF/GGUF vocabulary mismatch at ID {i}: {token!r} vs {actual!r}")

    def generate(self, prompt, history):
        formatted = prompt
        if self.key == "stentor" and not self.args.raw:
            formatted = self.template.render(messages=history + [{"role": "user", "content": prompt}],
                                             bos_token="<s>", eos_token="</s>", add_generation_prompt=True)
        ids = self.tokenizer.encode(formatted, add_special_tokens=False).ids
        if not ids or ids[0] != self.model.token_bos():
            ids.insert(0, self.model.token_bos())
        if len(ids) + self.args.max_tokens > self.info["context"]:
            raise ValueError(f"Input ({len(ids)}) + reply ({self.args.max_tokens}) exceeds "
                             f"{self.info['context']}; use /new or shorten input")
        # Original HF IDs go straight into llama.cpp; the GGUF tokenizer is not used.
        self.model.reset()
        generated = []
        stream = self.model.generate(ids, top_k=40, top_p=0.9, temp=self.args.temperature,
                                     repeat_penalty=1.1, reset=True)
        try:
            for token in stream:
                generated.append(token)
                if token == self.model.token_eos() or len(generated) >= self.args.max_tokens:
                    break
        finally:
            stream.close()
        return dict(answer=self.tokenizer.decode(generated, skip_special_tokens=True),
                    formatted_prompt=formatted, input_ids=ids, output_ids=generated,
                    input_tokens=len(ids), generated_tokens=len(generated))


class VaghRunner:
    def __init__(self, args):
        verify_local("vagh")
        import torch
        import sentencepiece as spm
        # Only these pinned, reviewed upstream modules are imported. No remote-code loader.
        sys.path.insert(0, str(STORE / "vagh"))
        from model.config import VaghLMConfig
        from model.transformer import VaghLM
        self.torch, self.args = torch, args
        torch.set_num_threads(args.threads)
        torch.manual_seed(42)
        self.config = VaghLMConfig()
        self.model = VaghLM(self.config)
        ckpt = torch.load(STORE / "vagh/checkpoints/best_checkpoint.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=str(STORE / "vagh/tokenizer/vocab/tokenizer.model"))
        self.info = dict(backend="torch", version=torch.__version__, dtype="float32", device="cpu",
                         parameters=self.model.count_parameters(), checkpoint="best_checkpoint.pt",
                         context=self.config.context_length, step=ckpt.get("step"), loss=ckpt.get("loss"))

    def generate(self, prompt, history):
        t = self.torch
        c = self.config
        ids = [c.bos_id]
        if self.args.raw:
            ids += self.sp.encode(prompt)
        else:
            for item in history:
                ids += [c.user_id if item["role"] == "user" else c.assistant_id]
                ids += self.sp.encode(item["content"])
                if item["role"] == "assistant":
                    ids += [c.eos_id]
            ids += [c.user_id] + self.sp.encode(prompt) + [c.assistant_id]
        if len(ids) + self.args.max_tokens > c.context_length:
            raise ValueError("Context is full; use /new or shorten input")
        x = t.tensor([ids], dtype=t.long)
        generated = []
        with t.inference_mode():
            for _ in range(self.args.max_tokens):
                logits = self.model(x)[0][:, -1, :]
                # Same repetition/top-k/top-p approach as the author's inference script.
                for token in set(x[0].tolist()):
                    logits[0, token] = (logits[0, token] / 1.3 if logits[0, token] > 0
                                        else logits[0, token] * 1.3)
                if self.args.temperature == 0:
                    nxt = t.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits /= self.args.temperature
                    values, _ = t.topk(logits, 40)
                    logits[logits < values[:, [-1]]] = -float("inf")
                    values, indices = t.sort(logits, descending=True)
                    probs = t.softmax(values, dim=-1)
                    values[probs.cumsum(dim=-1) - probs > 0.9] = -float("inf")
                    logits = t.full_like(logits, -float("inf")).scatter_(1, indices, values)
                    nxt = t.multinomial(t.softmax(logits, dim=-1), 1)
                token = nxt.item()
                generated.append(token)
                x = t.cat([x, nxt], dim=1)
                if token == c.eos_id:
                    break
        return dict(answer=self.sp.decode(generated), input_ids=ids, output_ids=generated,
                    input_tokens=len(ids), generated_tokens=len(generated))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fetch", "chat", "complete"])
    p.add_argument("model", choices=["all", *LOCK])
    p.add_argument("--prompt", default="Hello! My name is Sofia. What is your name?")
    p.add_argument("--raw", action="store_true", help="Plain text continuation, no chat formatting")
    p.add_argument("--history", action="store_true", help="Keep previous turns (default: each input independent)")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--trace", type=Path, help="New JSONL file with exact requests and outputs")
    a = p.parse_args()
    if not 1 <= a.max_tokens <= 256 or not 0 <= a.temperature <= 2 or a.threads < 1:
        p.error("max-tokens: 1..256; temperature: 0..2; threads >= 1")
    if a.mode == "fetch":
        for key in (LOCK if a.model == "all" else [a.model]):
            fetch(key)
        return
    if a.model == "all":
        p.error("Choose one model for chat/complete")
    if a.history and (a.raw or a.model == "stories"):
        p.error("--history is only for dialogue mode")
    if a.mode == "complete" and (not a.prompt.strip() or a.prompt.startswith("/")):
        p.error("complete needs a nonempty text prompt, not a console command")
    runner = VaghRunner(a) if a.model == "vagh" else GGUFRunner(a.model, a)
    print(f"{a.model}: {runner.info['parameters']:,} parameters, "
          f"{runner.info.get('quantization', 'FP32')}, CPU; context {runner.info['context']} tokens", flush=True)
    if a.model == "vagh":
        print("Published best_checkpoint.pt, FP32. The advertised finetuned_v2_best.pt is unavailable.")
    print("Independent inputs; no Ember persona." if not a.history else "History ON; /new clears it.")
    if a.model == "stories" or a.raw:
        print("Text continuation: type the beginning of an English story.")
    history = []
    trace = None
    try:
        if a.trace:
            a.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = a.trace.open("x")
        while True:
            if a.mode == "complete":
                prompt = a.prompt
            else:
                try:
                    prompt = input("You> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if prompt == "/quit":
                break
            if prompt == "/new":
                history.clear()
                print("New conversation.")
                continue
            if prompt == "/stats":
                print(json.dumps(runner.info), f"history_turns={len(history)//2}")
                continue
            if prompt == "/help":
                print("/new /stats /help /quit; Ctrl+C exits. --history enables dialogue memory.")
                continue
            if not prompt.strip():
                continue
            started = time.monotonic()
            try:
                result = runner.generate(prompt, history)
            except ValueError as error:
                print(error)
                if a.mode == "complete":
                    raise
                continue
            result.update(user=prompt, seconds=time.monotonic()-started, model=a.model,
                          revision=LOCK[a.model]["revision"], info=runner.info,
                          settings=dict(temperature=a.temperature, max_tokens=a.max_tokens,
                                        history=a.history, raw=a.raw, threads=a.threads))
            print(f"{a.model}> {result['answer'] or '[empty output]'}", flush=True)
            if trace:
                trace.write(json.dumps(result, ensure_ascii=False) + "\n")
                trace.flush()
            if a.history:
                history.extend([dict(role="user", content=prompt),
                                dict(role="assistant", content=result["answer"])])
            if a.mode == "complete":
                break
    finally:
        if trace:
            trace.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (OSError, RuntimeError, ImportError) as error:
        sys.exit(f"Error: {error}")
