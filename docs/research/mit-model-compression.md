# Bigger chat model → smaller dragon: MIT materials and limits

Checked 2026-10-17. This is a research proposal, not a measured new model.

## What the course actually covers

MIT 6.5940 **TinyML and Efficient Deep Learning Computing**, Fall 2024, separates pruning (lectures 3–4), quantization (5–6), knowledge distillation (9), MCU deployment (10–11), and efficient LLM deployment (13). Its LLM lab target is a **laptop running Llama2-7B**, not an ESP32. This matters when interpreting the word “edge.” [Official syllabus](https://hanlab.mit.edu/courses/2024-fall-65940).

Official slide links from that syllabus:

- [Lecture 3: Pruning I](https://www.dropbox.com/scl/fi/6qspcmk8qayy7mft737gh/Lec03-Pruning-I.pdf?dl=0&rlkey=9jpifc92be0sitiknpbhn9ggf).
- [Lecture 4: Pruning II](https://www.dropbox.com/scl/fi/w5baiyci5cxl1ozpy6lsr/Lec04-Pruning-II.pdf?dl=0&rlkey=6qxc1nz20isy9izwnqfebtukg).
- [Lecture 9: Knowledge Distillation](https://www.dropbox.com/scl/fi/fjgnue7z3mi1ynxbd0y5k/Lec09-Knowledge-Distillation.pdf?dl=0&rlkey=cup1qhlpx3vx0nrs7wuwj6m0d).
- [Lecture 13: Efficient LLM Deployment](https://www.dropbox.com/scl/fi/aa5ea0hrc68cn3fh18nan/Lec13-LLM-Deployment.pdf?dl=0&rlkey=gzq9yiddx4bnh14bxomtfmcoj).

PDF text from lectures 3, 9, and 13 was downloaded from those official links and extracted locally with macOS PDFKit. Relevant checked slides:

- Lecture 3, pp. 10–14: choose pruning granularity, criterion and ratio, then fine-tune the pruned model. Pages 29–30 contrast irregular fine-grained pruning with structured pruning yielding a smaller matrix; pp. 36–37 discuss the acceleration tradeoff.
- Lecture 9, pp. 4–7: tiny models can learn with help from a large model; the distillation diagram combines teacher/student logit matching with a task loss. Page 5 specifically illustrates underfitting by tiny vision models, not a universal LLM parameter threshold.
- Lecture 13, pp. 20–23: low-bit weight-only quantization targets the generation memory bottleneck and must account for important weight channels. Page 40 demonstrates a 7B model on Jetson Orin Nano with about **7 GB** available memory. That example is far above our ESP32 memory budget.

The quantitative research claims below are checked against the original papers. No claim here requires having watched the complete lecture videos.

## Compression methods are different operations

| Method | What changes | Implication for this project |
|---|---|---|
| Weight quantization | Fewer bits for each retained weight | Q4 helps storage, but does not turn 135M parameters into 20M parameters. |
| Structured pruning | Remove whole channels, heads, or layers; produce smaller dense tensors | Potentially compatible with a simple dense MCU runtime, followed by recovery training. |
| Unstructured pruning | Set selected individual weights to zero | Requires sparse storage and suitable kernels to realize savings; zeros in our existing dense arrays would not suffice. |
| Distillation | Train a smaller student to reproduce a teacher's behavior or probability distributions | Can recover a pruned model or train a different architecture; capacity and training data still limit the result. |

These distinctions follow the pruning/quantization pipeline in [Deep Compression](https://arxiv.org/abs/1510.00149) and the structured pruning/distillation formulation in [Minitron](https://arxiv.org/html/2407.14679v1). Runtime implications are our engineering interpretation for the current dense ESP implementation.

Song Han's **Deep Compression** reports 35× storage compression for AlexNet and 49× for VGG-16, combining pruning, trained quantization, and Huffman coding with retraining. Those are **ImageNet vision networks**, not evidence that a conversational LLM can lose 98% of its parameters without losing quality. The ratio also combines different kinds of compression; it is not a parameter-count ratio. [Original paper](https://arxiv.org/abs/1510.00149).

The HAN Lab **AWQ** work is low-bit, activation-aware weight quantization. It uses offline activation statistics to protect important channels and demonstrates GPU-oriented TinyChat implementations. It does not establish ESP32 throughput or remove most parameters. Our existing plain block Q4 converter is not AWQ. [AWQ paper](https://arxiv.org/abs/2306.00978).

## A real LLM example: Minitron

NVIDIA derives approximately **8B and 4B models from a 15B model**, using structured pruning and recovery distillation. The paper favors width pruning over depth at its studied scales, ranks candidates after some recovery, and recommends pruning from a model close to the target size. Its best-practice list distinguishes logit-only distillation from additional intermediate-state losses when depth changes substantially. [Sections 2–4](https://arxiv.org/html/2407.14679v1).

The recovery budget is substantial: the 8B result lists **94 billion training tokens**; even “lightweight” architecture comparisons use about **1.8 billion tokens**. “Up to 40× fewer training tokens” compares with training a model from scratch; it is not 40× parameter compression. These billion-parameter results support the method, but do not prove that **135M → 20M** will preserve chat quality or that our few hundred examples are sufficient. [Tables 1–3 and §4.1](https://arxiv.org/html/2407.14679v1).

## Our memory arithmetic

For an approximate weight budget, ideal Q4 uses `parameters / 2` bytes. Our block-32 representation adds a two-byte scale to each sixteen-byte block, giving `parameters × 18 / 32` bytes before unquantized tensors, headers, tokenizer, and firmware. These are calculations, not measured exports.

| Dense parameter count | Ideal Q4 | Block-32 Q4 estimate |
|---|---:|---:|
| 20M | 9.54 MiB | 10.73 MiB |
| 25M | 11.92 MiB | 13.41 MiB |
| SmolLM2-135M, approximately 134.5M | 64.14 MiB | 72.16 MiB |

With 16 MiB flash, 20–25M is an exploratory weight-budget range, **not a guaranteed deployable size**. Code, tokenizer, KV cache, activations, PSRAM traffic and latency must be measured separately. Weights may be memory-mapped from flash; the 8 MiB PSRAM figure is not the only storage limit.

There is a practical obstacle in **SmolLM2-135M-Instruct**: its tied embedding is `49,152 × 576 = 28,311,552` parameters. That matrix alone is about **15.19 MiB** in block-32 Q4. Removing transformer layers alone cannot reach a 20–25M total model. We would also need width reduction and/or vocabulary changes with recovery training, or a separately designed student. [Official config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json).

## Recommended bounded experiment

This recommendation is our synthesis, not an MIT guarantee:

1. Evaluate a stronger English chat model such as SmolLM2-135M-Instruct **locally and uncompressed first** on dragon dialogue, name binding, corrections, unseen names, and unknown facts. Its suitability remains untested here.
2. If the baseline is meaningfully better, test modest structured compression with teacher-based recovery. Preserve a common tokenizer initially so probability-distribution distillation is straightforward. Compare against an equally sized separately trained student where feasible.
3. Explore progressively smaller target sizes, measuring after recovery; stop when conversational quality collapses. This is a size/quality investigation, not a commitment that every stage will work. Depth-only pruning cannot solve the final memory budget above.
4. Only for a student with useful unquantized behavior, add calibrated Q4 evaluation, calculate the real firmware footprint, and measure on the ESP32.

Compared with our current pilot, the difference would be **inheriting and recovering a better conversational model**, rather than expecting a small story model to acquire general dialogue competence from 627 short teacher answers. Our existing experiments used generated-text SFT; they did not implement the logit-based recovery evaluated in Minitron. See [local experiment report](../dragon-training.md).
