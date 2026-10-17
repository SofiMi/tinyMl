#!/usr/bin/env python3
"""Freeze the reviewed pilot data separately from raw teacher output."""
import argparse
import hashlib
import json
import re
from pathlib import Path

# Reviewed on the raw Qwen3.5-9B pilot; retain the reasons for reproducibility.
EXCLUDE = {
    "7eb7eb0d7fd3141b": "Invents another friend named Grace after a name correction.",
    "ff3035e8650d59bd": "Invents a tail for the human user.",
    "ff2fd8727035a289": "Implies the user's name could establish their school.",
    "cf85ce44e7e58732": "Unnecessary negative comparison with the user's old name.",
    "0ac877198e721188": "Confuses the user's corrected name with the dragon's name.",
    "11ab8d87d44bb5a0": "Confuses the user's corrected name with the dragon's name.",
    "a125490c6b2acc28": "Fails to answer a simple question about ice.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("build/dragon/data"))
    p.add_argument("--output", type=Path, default=Path("build/dragon/pilot-data"))
    p.add_argument("--review", type=Path, help="JSON mapping of conversation IDs to rejection reasons; overrides original-pilot review")
    args = p.parse_args()
    exclude = EXCLUDE if args.review is None else json.loads(args.review.read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())
    seen = set()
    for split in ["train", "validation", "rejected"]:
        path = args.input / f"{split}.jsonl"
        if path.exists():
            seen.update(json.loads(x)["id"] for x in path.read_text().splitlines())
    if seen != {r["id"] for r in manifest["scenarios"]}:
        raise SystemExit("Raw generation is incomplete; finish it before freezing the pilot")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"teacher_manifest": manifest, "removed": [], "files": {}}
    for split in ["train", "validation"]:
        rows = [json.loads(x) for x in (args.input / f"{split}.jsonl").read_text().splitlines()]
        kept = []
        for row in rows:
            reason = exclude.get(row["id"])
            if row["family"] == "preference":
                reason = "Question asks for a favorite color, but the premise only says the user likes it."
            if row["family"] in {"name", "name_variation", "separate_names", "correction", "correction_variation", "validation_names"}:
                answer = row["messages"][-1]["content"]
                for name in row["expected"]:
                    if name != "ember" and re.search(r"\b(?:my name is|you can call me|call me|i am|i'm)\s+" + re.escape(name) + r"\b", answer, re.I):
                        reason = "Teacher uses the user's name as its own in the final answer."
            if reason:
                report["removed"].append({"id": row["id"], "reason": reason})
            else:
                kept.append(row)
        content = "".join(json.dumps(r) + "\n" for r in kept)
        path = args.output / f"{split}.jsonl"
        path.write_text(content)
        report["files"][split] = {"conversations": len(kept), "answers": sum(len(r["messages"])//2 for r in kept),
                                 "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["files"], indent=2))


if __name__ == "__main__":
    main()
