#!/usr/bin/env python3
"""Count real target tokens and context needs; loads a tokenizer, never a model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="local SentencePiece .model file")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    report = {"tokenizer": str(args.tokenizer), "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "protocol": "stories15M: BOS, User/Dragon, assistant EOS; fixed persona omitted from prompt",
        "warning": "Sizing only, not a selected/trained student or proof of ESP compatibility. Export contains a system message; the legacy trainer needs an explicit fixed-persona conversion. Actual pruned TinyTalk tokenizer differs.", "splits": {}}
    for split in ("train", "validation", "test"):
        path = args.data / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        lengths, targets, contexts, final_prompts = [], [], [], []
        for row in rows:
            messages = row["messages"]
            assert messages[0]["role"] == "system"
            history = [sp.bos_id()]
            count = 0
            for i in range(1, len(messages), 2):
                user, answer = messages[i:i+2]
                assert user["role"] == "user" and answer["role"] == "assistant"
                history += sp.encode("User: " + user["content"] + "\nDragon:")
                prefix_length = len(history)
                target = sp.encode(" " + answer["content"]) + [sp.eos_id()]
                targets.append(len(target))
                count += len(target)
                contexts.append(prefix_length + len(target))
                history += target + sp.encode("\n")
            lengths.append(count)
            if len(messages) > 1:
                final_prompts.append(prefix_length)
        report["splits"][split] = {"conversations": len(rows), "answers": len(targets),
            "supervised_tokens_per_epoch": sum(targets),
            "mean_target_tokens_per_answer": statistics.mean(targets) if targets else None,
            "mean_target_tokens_per_conversation": statistics.mean(lengths) if lengths else None,
            "all_history_context_max": max(contexts, default=0),
            "answers_exceeding_256_with_full_history": sum(n > 256 for n in contexts),
            "answers_exceeding_512_with_full_history": sum(n > 512 for n in contexts),
            "last_prompts_exceeding_72_minus_8_runtime_budget": sum(n > 64 for n in final_prompts),
            "projected_supervised_tokens_for_n_conversations": {str(n): round(n*statistics.mean(lengths)) for n in (10000, 50000, 150000)} if lengths else {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
