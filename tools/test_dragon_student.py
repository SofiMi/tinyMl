"""Check loss alignment and context handling with the real student tokenizer."""
import json
import tempfile
import unittest
from pathlib import Path
import sentencepiece as spm
from dragon_student import examples, fit_prompt, prompt_tokens, batch, q4_roundtrip
import torch


class TrainingFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.sp = spm.SentencePieceProcessor(model_file=os.environ["DRAGON_TOKENIZER"])

    def test_only_answers_and_eos_are_targets(self):
        row = {"messages": [
            {"role": "user", "content": "Call me Ada."},
            {"role": "assistant", "content": "Hello Ada!"},
            {"role": "user", "content": "Who am I?"},
            {"role": "assistant", "content": "You are Ada."},
        ]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            p.write_text(json.dumps(row) + "\n")
            data = examples(p, self.sp, 256)
        for index, (inputs, labels) in enumerate(data):
            history = [] if index == 0 else [("Call me Ada.", "Hello Ada!")]
            question = row["messages"][index * 2]["content"]
            answer = row["messages"][index * 2 + 1]["content"]
            prefix = prompt_tokens(self.sp, history, question)
            self.assertEqual(inputs[:len(prefix)], prefix)
            self.assertEqual(labels[:len(prefix)-1], [-1] * (len(prefix)-1))
            self.assertEqual(labels[len(prefix)-1:], self.sp.encode(" " + answer) + [self.sp.eos_id()])
            self.assertEqual(len(inputs), len(labels))

    def test_trim_whole_old_turns_and_keep_question(self):
        question = "Can we play?"
        keep = [("Hi!", "Hello!")]
        budget = len(prompt_tokens(self.sp, keep, question))
        history = [("Old " * 100, "Long " * 100)] + keep
        self.assertEqual(fit_prompt(self.sp, history, question, budget), prompt_tokens(self.sp, keep, question))
        with self.assertRaises(ValueError):
            fit_prompt(self.sp, [], "word " * 300, 12)

    def test_padding_does_not_add_loss_targets(self):
        x, y = batch([([1, 2, 3], [-1, 4, 2]), ([1], [2])], "cpu")
        self.assertEqual(x.tolist(), [[1, 2, 3], [1, 0, 0]])
        self.assertEqual(y.tolist(), [[-1, 4, 2], [2, -1, -1]])

    def test_q4_zero_and_representable_weights(self):
        zero = torch.zeros(2, 32)
        self.assertTrue(torch.equal(q4_roundtrip(zero), zero))
        exact = torch.arange(-8, 8, dtype=torch.float32).repeat(2).reshape(1, 32)
        self.assertTrue(torch.equal(q4_roundtrip(exact), exact))


if __name__ == "__main__":
    unittest.main()
