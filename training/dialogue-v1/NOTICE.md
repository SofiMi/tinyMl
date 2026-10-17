# Data provenance

The general conversation subset is derived from HuggingFaceTB/smol-smoltalk,
revision f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc. Its dataset card declares
Apache-2.0. Original source labels and shard row indices are retained in JSONL;
filtering, splitting and the added neutral system message are local modifications.
See https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk and the source
datasets referenced there. Credit: Loubna Ben Allal et al., SmolLM2 (2025).

The remaining rows were generated locally from deterministic factual templates
in tools/prepare_dialogue_data.py. No API or new local-teacher generation was used.
Provenance and content hashes are recorded in provenance.json.
