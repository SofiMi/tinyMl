"""Check HF prompt boundaries with the real locally downloaded tokenizer."""
import os
import unittest
from pathlib import Path
from transformers import AutoTokenizer
from dragon_hf import fit_messages


class ChatFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("DRAGON_HF_MODEL", str(Path.home()/".cache/tinyml-smollm2-135m-instruct"))
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def test_chatml_roles_and_generation_boundary(self):
        messages = [{"role": "system", "content": "You are Ember."},
                    {"role": "user", "content": "Hello!"}]
        ids = fit_messages(self.tokenizer, messages, 200)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertEqual(self.tokenizer.decode(ids),
                         "<|im_start|>system\nYou are Ember.<|im_end|>\n"
                         "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")

    def test_trimming_preserves_system_newest_pairs_and_question(self):
        system = {"role": "system", "content": "You are Ember."}
        recent = [{"role": "user", "content": "I am Ada."},
                  {"role": "assistant", "content": "Hello Ada."},
                  {"role": "user", "content": "Who am I?"}]
        expected = fit_messages(self.tokenizer, [system] + recent, 1000)
        history = [system, {"role": "user", "content": "Old " * 100},
                   {"role": "assistant", "content": "Reply " * 100}] + recent
        self.assertEqual(fit_messages(self.tokenizer, history, len(expected)), expected)
        self.assertEqual(len(history), 6)
        with self.assertRaises(ValueError):
            fit_messages(self.tokenizer, [system, recent[-1]], 3)


if __name__ == "__main__":
    unittest.main()
