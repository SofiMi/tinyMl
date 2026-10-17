"""Local GPT-Neo/TinyTalk adapter for the same distillation experiment."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM


class NeoStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        config._attn_implementation = "eager"
        self.hf = GPTNeoForCausalLM(config)
        self.params = SimpleNamespace(max_seq_len=256)
        self.last_loss = None

    @property
    def tok_embeddings(self):
        return self.hf.get_input_embeddings()

    def forward(self, tokens, targets=None):
        logits = self.hf(input_ids=tokens, use_cache=False).logits
        self.last_loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)
        return logits if targets is not None else logits[:, -1:]


class NeoTokenizer:
    bot_role = "Bot"

    def __init__(self, directory):
        self.hf = AutoTokenizer.from_pretrained(directory, local_files_only=True)

    def encode(self, text):
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.hf.decode(tokens, skip_special_tokens=True)

    def bos_id(self):
        return self.hf.eos_token_id

    def eos_id(self):
        return self.hf.eos_token_id


def initial_model(directory, checkpoint):
    from safetensors.torch import load_file
    config = GPTNeoConfig.from_pretrained(directory, local_files_only=True)
    model = NeoStudent(config)
    state = load_file(str(checkpoint))
    # The source safetensors omits the shared output embedding.
    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["transformer.wte.weight"]
    model.hf.load_state_dict(state, strict=True)
    return model, {"kind": "gpt_neo", "config": config.to_dict()}
