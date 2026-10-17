#!/usr/bin/env python3
"""Prepare an untrained natural-dialogue draft; preserve greetings and distinct conversations."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from prepare_dialogue_data import NEUTRAL, REVISION


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select_conversation(messages):
    """Keep up to three complete short exchanges; require actual multi-turn content."""
    pairs = []
    for i in range(0, min(len(messages)-1, 6), 2):
        q, a = messages[i:i+2]
        if q["role"] != "user" or a["role"] != "assistant":
            break
        if any(not 1 <= len(m["content"].split()) <= 64 or "```" in m["content"] for m in [q, a]):
            break
        pairs.extend([q, a])
    if len(pairs) < 4:
        return None
    substantive = next((m["content"] for m in pairs[::2] if len(m["content"].split()) >= 5), None)
    if substantive is None:
        return None
    # Full-conversation deduplication; split group ignores short opening greetings.
    identifier = key(json.dumps(pairs, sort_keys=True, ensure_ascii=False))
    group = key(" ".join(substantive.lower().split()))
    return identifier, group, pairs


def main():
    shard = Path.home()/".cache/tinyml-smol-smoltalk/data/train-00000-of-00004.parquet"
    selected = {s: {source: [] for source in ["everyday-conversations", "smol-magpie-ultra-short"]} for s in ["train", "validation"]}
    seen = set()
    for index, row in enumerate(pq.read_table(shard).to_pylist()):
        source = row["source"]
        if source not in selected["train"]:
            continue
        result = select_conversation(row["messages"])
        if result is None:
            continue
        identifier, group, pairs = result
        if identifier in seen:
            continue
        seen.add(identifier)
        split = "validation" if int(group[:8], 16) % 10 == 0 else "train"
        selected[split][source].append({"id": identifier, "split_group": group, "source": source,
            "shard_row": index, "family": "general", "messages": [{"role": "system", "content": NEUTRAL}]+pairs})
    out = Path("training/dialogue-v2-natural-draft")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    report = {"status": "data draft only; no model trained on these files",
              "source": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk", "revision": REVISION,
              "license": "upstream Apache-2.0; see ../dialogue-v1/NOTICE.md and LICENSE-APACHE-2.0",
              "shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "seed": 42,
              "limitations": "Automatic length/source filter, no full semantic review. Split groups the first substantive user question, not all paraphrases; a separate scenario-held-out evaluation is still required.",
              "splits": {}}
    groups = {}
    for split, cap in [("train", 1500), ("validation", 170)]:
        rows = selected[split]["everyday-conversations"][:]
        rest = selected[split]["smol-magpie-ultra-short"]
        rng.shuffle(rest)
        rows += rest[:cap]
        rng.shuffle(rows)
        groups[split] = {r["split_group"] for r in rows}
        content = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)
        (out/f"{split}.jsonl").write_text(content)
        report["splits"][split] = {"rows": len(rows), "answers": sum((len(r["messages"])-1)//2 for r in rows),
                                  "sources": dict(Counter(r["source"] for r in rows)), "sha256": key(content)}
    assert not groups["train"] & groups["validation"]
    (out/"provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
