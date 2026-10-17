#!/usr/bin/env python3
"""Evaluate pinned original Doge code under isolated Transformers 4.48.3."""
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import dragon_hf

REVISION = "dc9b359e279815619616f9b08d4c3e26d0d7dc0c"
CODE = {
    "configuration_doge.py": "54543d8bede7832fa60a777912cf3cb06666591f53f5b53fc24377e9768a5ec2",
    "modeling_doge.py": "f80d630c7bd00a89a9f340d05a99fe6766f8c32640ab0b0c6e0dd90315a791e0",
}


def load_legacy(path, device):
    if transformers.__version__ != "4.48.3" or device != "cpu":
        raise RuntimeError("Use isolated Transformers 4.48.3 and CPU; see reconsideration report")
    for name, expected in CODE.items():
        if dragon_hf.sha(path/name) != expected:
            raise RuntimeError(f"Unreviewed legacy code: {name}")
    config = json.loads((path/"config.json").read_text())
    if config.get("auto_map") != {"AutoConfig": "configuration_doge.DogeConfig", "AutoModelForCausalLM": "modeling_doge.DogeForCausalLM"}:
        raise RuntimeError("Unexpected dynamic code mapping")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise RuntimeError(f"Checkpoint did not load exactly: {info}")
    info = {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}
    model.eval()
    ids = tokenizer("Hello, my name is Sofia.", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(full, cached, atol=1e-4, rtol=1e-4)
    audit = {"revision": REVISION, "code_sha256": CODE, "transformers": transformers.__version__,
             "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
             "loading_info": info, "cache_max_abs_difference": (full-cached).abs().max().item(),
             "cache_check": "last-token full forward vs prefix-cache incremental forward, atol=rtol=1e-4"}
    (path/"legacy-validation.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit), flush=True)
    return model, tokenizer


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("loop", "natural"):
        raise SystemExit("Usage: doge_legacy_eval.py loop|natural [evaluation options]")
    mode = sys.argv.pop(1)
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", str(Path.home()/".cache/tinyml-doge-20m-instruct")])
    if mode == "loop":
        import diagnose_dialogue_loop
        diagnose_dialogue_loop.load = load_legacy
        diagnose_dialogue_loop.main()
    else:
        dragon_hf.load = load_legacy
        sys.argv.insert(1, "eval")
        dragon_hf.main()
