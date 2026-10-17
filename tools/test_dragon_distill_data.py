import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import dragon_distill_data as data


TURNS = [
    {"user": "Hi", "assistant": "Hello there, little friend!"},
    {"user": "I drew a bird today.", "assistant": "What color did you choose for its wings?"},
    {"user": "I used blue. Do you like it?", "assistant": "Blue wings sound lovely to me."},
    {"user": "I must go now.", "assistant": "See you soon, my friend!"},
]


class DistillationDataTests(unittest.TestCase):
    def test_splits_are_by_scenario_before_wording(self):
        rows = data.make_plan(2000, 42)
        self.assertEqual(rows, data.make_plan(2000, 42))
        groups = {}
        for row in rows:
            groups.setdefault(row["split_group"], set()).add(row["split"])
            if row["user_name"]:
                self.assertIn(row["user_name"], data.NAMES[row["split"]])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})
        counts = data.Counter(r["family"] for r in rows)
        self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        self.assertEqual(len({r["id"] for r in rows}), 2000)

    def test_keep_one_word_greetings_and_reject_role_swaps(self):
        row = data.make_plan(1, 42)[0]
        self.assertEqual(data.validate(TURNS, row), [])
        row.update(user_name="Ada", must_introduce=["Ada"], expected_last=["Ada"])
        turns = json.loads(json.dumps(TURNS))
        turns[0]["user"] = "My name is Ada."
        turns[-1]["assistant"] = "Your name is Ada."
        self.assertEqual(data.validate(turns, row), [])
        turns[-1]["assistant"] = "I am Ada. You are Ember."
        errors = data.validate(turns, row)
        self.assertIn("assistant adopts user's name", errors)
        self.assertIn("assistant assigns dragon name to human", errors)
        self.assertFalse(data.has_word("Your name is Adaline.", "Ada"))

    def test_reject_corruption_repetition_and_old_names(self):
        row = data.make_plan(1, 42)[0]
        self.assertTrue(data.validate(TURNS[:3], row))
        turns = json.loads(json.dumps(TURNS))
        turns[-1]["assistant"] = turns[0]["assistant"]
        self.assertIn("duplicate answer within dialogue", data.validate(turns, row))
        turns[-1]["assistant"] = "You are Ada, formerly Bea."
        row.update(old_name="Bea")
        self.assertIn("old name in final answer", data.validate(turns, row))
        turns[-1]["assistant"] = "Hello there! 🐉"
        self.assertTrue(data.validate(turns, row))

    def test_journal_export_and_resume_do_not_regenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan = {"version": data.VERSION, "seed": 42, "rows": data.make_plan(1, 42)}
            data.write_json(directory / "plan.json", plan)
            args = SimpleNamespace(output=directory, url="http://127.0.0.1:11434",
                                   model="test-teacher", limit=1, min_free_mb=64)
            calls = []

            def request(base, endpoint, payload=None):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test-teacher", "digest": "sha256:test"}]}
                calls.append(payload)
                result = {"turns": TURNS} if len(calls) == 1 else {"pass": True, "issues": []}
                return {"message": {"content": json.dumps(result)}, "done_reason": "stop", "eval_count": 20}

            with patch.object(data, "request", request), contextlib.redirect_stdout(io.StringIO()):
                data.generate(args)
                data.generate(args)
            self.assertEqual(len(calls), 2)  # generator and critic, none on resume
            self.assertEqual(len(data.journal_rows(directory)), 1)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["accepted"], 1)
            split = plan["rows"][0]["split"]
            record = json.loads((directory / f"{split}.jsonl").read_text())
            self.assertEqual([m["role"] for m in record["messages"]], ["system"] + ["user", "assistant"]*4)
            self.assertEqual(record["messages"][1]["content"], "Hi")

    def test_critic_failure_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data.write_json(directory / "plan.json", {"version": data.VERSION, "seed": 42, "rows": data.make_plan(1, 42)})
            args = SimpleNamespace(output=directory, url="http://localhost:11434", model="test", limit=1, min_free_mb=64)

            def request(base, endpoint, payload=None):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test", "digest": "sha256:test"}]}
                obj = {"turns": TURNS} if payload["messages"][0]["content"] == data.GENERATOR else {"pass": False, "issues": ["contradiction"]}
                return {"message": {"content": json.dumps(obj)}, "done_reason": "stop", "eval_count": 20}

            with patch.object(data, "request", request), contextlib.redirect_stdout(io.StringIO()):
                data.generate(args)
            self.assertEqual(data.export(directory)["accepted"], 0)
            self.assertEqual((directory / "train.jsonl").read_text(), "")

    def test_manual_quarantine_overrides_automatic_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            row = data.make_plan(1, 42)[0]
            record = {**row, "turns": TURNS, "status": "accepted", "calls": [],
                      "wall_seconds": 1, "reasons": []}
            (directory / "journal.jsonl").write_text(json.dumps(record)+"\n")
            data.write_json(directory / "manual-review.json", {row["id"]: {"pass": False, "reason": "bad fact"}})
            result = data.export(directory)
            self.assertEqual(result["automatic_passes"], 1)
            self.assertEqual(result["accepted"], 0)
            self.assertEqual(result["manual_rejections"], 1)
            self.assertFalse(any((directory / f"{s}.jsonl").read_text() for s in ("train", "validation", "test")))

    def test_fact_prompts_preserve_human_question_direction(self):
        rows = data.make_plan(120, 42)
        for row in rows:
            if row["family"] in ("identity", "correction", "friend", "preference", "ownership", "unknown"):
                self.assertEqual(len(row["human_prompts"]), 4)
                turns = [{"user": q, "assistant": "I can answer that."} for q in row["human_prompts"]]
                turns[-1]["user"] = "What is your name?"
                self.assertIn("user prompts changed", data.validate(turns, row))

    def test_local_only(self):
        for url in ("https://example.com", "http://192.168.1.2:11434", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                data.local_url(url)

    def test_fixed_questions_include_actual_previous_teacher_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            row = data.make_plan(12, 42)[6]
            data.write_json(directory / "plan.json", {"version": data.VERSION, "seed": 42, "rows": [row]})
            args = SimpleNamespace(output=directory, url="http://localhost:11434", model="test", limit=1, min_free_mb=64)
            answers = ["Hello, nice to meet you!", "I enjoy doing that with friends.",
                       "My wings help me fly.", f"I am Ember, and you are {row['user_name']}."]
            calls = []

            def request(base, endpoint, payload=None):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test", "digest": "sha256:test"}]}
                calls.append(json.loads(json.dumps(payload)))
                number = len(calls)-1
                answer = answers[number] if number < 4 else json.dumps({"pass": True, "issues": []})
                return {"message": {"content": answer}, "done_reason": "stop", "eval_count": 10}

            with patch.object(data, "request", request), contextlib.redirect_stdout(io.StringIO()):
                data.generate(args)
            self.assertEqual(len(calls), 5)
            for number in range(4):
                messages = calls[number]["messages"]
                self.assertEqual(messages[-1]["content"], row["human_prompts"][number])
                self.assertEqual([m["content"] for m in messages if m["role"] == "assistant"], answers[:number])
            self.assertEqual(data.export(directory)["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
