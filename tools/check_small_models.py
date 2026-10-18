#!/usr/bin/env python3
"""Real CLI/inference checks; these verify plumbing, not conversation quality."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from small_models import STORE, LOCK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Local computer inference; not ESP and not a quality benchmark", "runs": []}
    for key in ("stentor", "stories", "vagh"):
        questions = (["Once upon a time, a baby dragon met a little boy named Ramil.",
                      "The little dragon was afraid of the dark. One night,"] if key == "stories" else
                     ["Hello! My name is Sofia. What is your name?", "What color is grass?"])
        path = args.output / f"{key}.jsonl"
        cmd = [sys.executable, str(ROOT / "tools/small_models.py"), "chat", key,
               "--max-tokens", "64", "--trace", str(path)]
        result = subprocess.run(cmd, input="\n".join([questions[0], "/stats", "/new", questions[1], "/quit"]) + "\n",
                                text=True, capture_output=True, timeout=180)
        (args.output / f"{key}.txt").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["user"] for r in rows] == questions
        assert "history_turns=0" in result.stdout and "New conversation." in result.stdout
        if key == "vagh":
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(STORE / "vagh/tokenizer/vocab/tokenizer.model"))
            for question, row in zip(questions, rows):
                assert row["input_ids"] == [2, 4] + sp.encode(question) + [5]
        else:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(STORE / key / "tokenizer.json"))
            for question, row in zip(questions, rows):
                expected = "<s>[INST] " + question + "[/INST]" if key == "stentor" else question
                assert row["formatted_prompt"] == expected
                ids = tokenizer.encode(expected, add_special_tokens=False).ids
                if ids[0] != 1:
                    ids.insert(0, 1)
                assert row["input_ids"] == ids
                assert row["input_tokens"] == len(ids)
                assert 0 < row["generated_tokens"] <= 64
        report["runs"].append(dict(model=key, revision=LOCK[key]["revision"], status="PASS",
                                   outputs=[r["answer"] for r in rows]))
        print(f"PASS {key}: real replies, exact independent prompts, /new, /stats, /quit", flush=True)
    for key in ("stentor", "vagh"):
        path = args.output / f"{key}-history.jsonl"
        cmd = [sys.executable, str(ROOT / "tools/small_models.py"), "chat", key,
               "--history", "--max-tokens", "16", "--trace", str(path)]
        questions = ["My name is Sofia.", "What is my name?", "Hello again!"]
        result = subprocess.run(cmd, input="\n".join([questions[0], questions[1], "/stats", "/new",
                                                     "word " * 3000, questions[2], "/quit"]) + "\n",
                                text=True, capture_output=True, timeout=180)
        (args.output / f"{key}-history.txt").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["user"] for r in rows] == questions
        assert "history_turns=2" in result.stdout
        assert "shorten input" in result.stdout
        if key == "stentor":
            expected = ("<s>[INST] " + questions[0] + "[/INST] " + rows[0]["answer"].strip()
                        + "</s>[INST] " + questions[1] + "[/INST]")
            assert rows[1]["formatted_prompt"] == expected
            assert rows[2]["formatted_prompt"] == "<s>[INST] " + questions[2] + "[/INST]"
        else:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(STORE / "vagh/tokenizer/vocab/tokenizer.model"))
            expected = [2, 4] + sp.encode(questions[0]) + [5] + sp.encode(rows[0]["answer"]) + [3, 4] + sp.encode(questions[1]) + [5]
            assert rows[1]["input_ids"] == expected
            assert rows[2]["input_ids"] == [2, 4] + sp.encode(questions[2]) + [5]
        report["runs"].append(dict(model=key, mode="history", status="PASS"))
        print(f"PASS {key}: own replies retained, /new resets, long input rejected without truncation", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
