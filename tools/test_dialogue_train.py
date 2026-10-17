import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from dialogue_train import examples, batch, reference_kl
from prepare_dialogue_data import controlled, NAMES


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(Path.home()/".cache/tinyml-minimind2-small", local_files_only=True)

    def test_causal_loss_and_padding(self):
        rows = [{"id": "a", "family": "test", "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "My name is Sofia."},
            {"role": "assistant", "content": "Hello, Sofia!"},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": "Sofia."}]}]
        samples, skipped = examples(rows, self.tok, 256)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["labels"][-1], self.tok.eos_token_id)
        target = [x for x in samples[1]["labels"] if x != -100]
        self.assertEqual(self.tok.decode(target, skip_special_tokens=True), "Sofia.")
        b = batch(samples, self.tok.pad_token_id, "cpu")
        self.assertTrue(torch.all(b["labels"][b["attention_mask"] == 0] == -100))
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(self.tok), hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
        out = model(**b)
        expected = F.cross_entropy(out.logits[:, :-1].reshape(-1, len(self.tok)), b["labels"][:, 1:].reshape(-1))
        torch.testing.assert_close(out.loss, expected)

    def test_controlled_splits_and_role(self):
        train = controlled(42, 100, NAMES[:32], "train")
        val = controlled(43, 30, NAMES[32:], "validation")
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        for row in train+val:
            self.assertNotIn("dragon", row["messages"][0]["content"])
            if row["family"] == "two_names":
                bot = row["messages"][0]["content"].split()[2].rstrip(",")
                user = row["messages"][1]["content"].split()[-1].rstrip(".")
                self.assertEqual(row["messages"][-1]["content"], f"I am {bot}. You are {user}.")

    def test_reference_kl_masks_and_direction(self):
        teacher = torch.tensor([[[2., 0.], [0., 2.], [0., 0.]], [[0., 2.], [2., 0.], [0., 0.]]])
        student = torch.zeros_like(teacher, requires_grad=True)
        labels = torch.tensor([[-100, 1, -100], [-100, 1, 1]])
        general = torch.tensor([True, False])
        loss = reference_kl(student, teacher, labels, general)
        p = teacher[0, 0].softmax(-1)
        expected = (p*(p.log()-student[0, 0].log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(student.grad[1].abs().sum()), 0.)
        self.assertEqual(float(student.grad[0, 1:].abs().sum()), 0.)
        torch.testing.assert_close(reference_kl(teacher, teacher, labels, general), torch.tensor(0.))


if __name__ == "__main__":
    unittest.main()
