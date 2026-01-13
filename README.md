# Update: Byte‑Level, Tokenizer‑Free Engram + Hugging Face Integration

This update introduces a **byte‑level, tokenizer‑free implementation of Engram** that is fully integrated with the Hugging Face (`transformers`) ecosystem. You can prepend this section to the original Engram README to describe the new code and how to use it.

---

## What This Fork Adds

This fork keeps the original **Engram idea**—a large static N‑gram memory table with **O(1) deterministic lookup** and **context‑aware gating**—but changes *where* N‑grams live and *how* the model is exposed:

1. **No tokenizer required**
   - The model works directly on **UTF‑8 bytes** (`0–255`), not BPE/WordPiece tokens.
   - N‑grams are built over **bytes** instead of tokens (e.g., 3‑gram, 4‑gram byte windows).
   - This avoids all tokenizer versioning / compatibility issues.

2. **Byte‑level Engram module**
   - For each position, the model takes the local byte N‑gram, hashes it, and uses the hash as an index into a **large trainable memory table**.
   - The retrieved memory vector is fused with the Transformer hidden state via a **learned gate**:
     \[
     \text{gate} = \sigma(\text{MLP}([h, m])),\quad
     h' = \text{gate}\cdot h + (1-\text{gate})\cdot m
     \]
   - This preserves the original Engram design: static pattern memory + dynamic, context‑aware usage.

3. **Full Hugging Face compatibility**
   - The model is implemented as a standard `PreTrainedModel` with a custom `PretrainedConfig`.
   - You can:
     - `save_pretrained` / `from_pretrained`
     - register it with `AutoConfig` / `AutoModelForCausalLM`
     - call `model.generate` as with any Causal LM
   - A minimal **byte‑level tokenizer** (`ByteTokenizer`) is provided so you can plug the model into HF `pipeline("text-generation")` without any extra files.

---

## High‑Level Architecture

The new implementation defines a byte‑level Causal LM with an Engram‑like memory block:

1. **Config – `CharEngramConfig`**
   - Inherits from `transformers.PretrainedConfig`.
   - Key fields:
     - `ngram_size`: byte N for N‑gram (e.g., 3 or 4)
     - `memory_dim`: dimension of each memory vector
     - `memory_capacity`: size of the memory table (number of hash buckets)
     - `hidden_size`, `num_layers`, `max_seq_len`: Transformer backbone hyperparameters
     - `engram_layer_index`: index of the layer after which the Engram module is inserted

2. **Tokenizer – `ByteTokenizer` (virtual, tokenizer‑free)**
   - Converts text to UTF‑8 bytes:
     - `encode(text) -> List[int]  # values in [0, 255]`
   - Converts byte IDs back to text:
     - `decode(List[int]) -> str`
   - `__call__` emulates HF tokenizers, returning:
     - `input_ids: LongTensor[B, L]`
     - `attention_mask: LongTensor[B, L]`

3. **Backbone – byte‑level Transformer**
   - `ByteEmbedding`: maps each byte (0–255) to an embedding and adds positional encoding.
   - `CharTransformerBlock`: basic Transformer encoder block (multi‑head self‑attention + FFN + LayerNorm), using `[B, L, H]` layout.
   - Stacks `num_layers` blocks; Engram is injected after `engram_layer_index`.

4. **Engram‑like memory – byte N‑grams**
   - `HashMapping`:
     - Takes a contiguous byte N‑gram (e.g., 4 bytes), converts it to a **deterministic hash**, and maps it to `[0, memory_capacity)`.
   - `ByteMemory`:
     - A big trainable table: `memory_table[memory_capacity, memory_dim]`.
     - For each position, computes the hash of the local N‑gram and returns the corresponding memory vector.
   - `GateFusion`:
     - Concatenates hidden state `h` and memory `m`, runs a small MLP + sigmoid to get a gate, and computes:
       - `out = gate * h + (1 - gate) * m`
     - Only the valid prefix (`L - ngram_size + 1`) is fused; the rest of the sequence is left unchanged.

5. **Model – `CharLevelEngramModel`**
   - Inherits `PreTrainedModel` and `GenerationMixin`.
   - Forward:
     - `input_ids` (bytes) → `ByteEmbedding` → Transformer layers
     - After `engram_layer_index`, run `CharEngram` (memory lookup + gated fusion)
     - Project final hidden states to 256‑dim byte vocabulary via `lm_head`
   - Output:
     - `logits: [B, L, vocab_size]` with `vocab_size = 256` (byte‑level LM)

---

## How to Use

### 1. Basic usage

```python
from char_engram_modeling import CharEngramConfig, CharLevelEngramModel, ByteTokenizer
import torch

# 1) Instantiate config and model
config = CharEngramConfig(
    ngram_size=4,
    hidden_size=256,
    num_layers=4,
    memory_capacity=100_000,
    max_seq_len=128,
)
model = CharLevelEngramModel(config)

# 2) Prepare input (no real tokenizer, just bytes)
tokenizer = ByteTokenizer()
batch = tokenizer("DeepSeek Engram is")

input_ids = batch["input_ids"]        # [1, L]
# (optionally move to GPU)
# input_ids = input_ids.to("cuda")
# model = model.to("cuda")

# 3) Forward pass (logits over bytes)
with torch.no_grad():
    outputs = model(input_ids)
logits = outputs.logits               # [1, L, 256]
```

### 2. Text generation (HF style)

```python
from char_engram_modeling import CharEngramConfig, CharLevelEngramModel, ByteTokenizer

config = CharEngramConfig()
model = CharLevelEngramModel(config)
tokenizer = ByteTokenizer()

prompt = "DeepSeek Engram is"
inputs = tokenizer(prompt)

with torch.no_grad():
    generated = model.generate(
        inputs["input_ids"],
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        pad_token_id=0,   # byte 0 is used as pad
    )

text = tokenizer.decode(generated[0].tolist())
print(text)
```

### 3. Save and load with Hugging Face format

```python
from char_engram_modeling import CharEngramConfig, CharLevelEngramModel

config = CharEngramConfig()
model = CharLevelEngramModel(config)

# Save in HF format
model.save_pretrained("./char_engram_model")

# Load later
loaded_model = CharLevelEngramModel.from_pretrained("./char_engram_model")
```

### 4. Register with `AutoConfig` / `AutoModelForCausalLM`

If you want to load the model via `AutoModelForCausalLM`:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from char_engram_modeling import CharEngramConfig, CharLevelEngramModel, ByteTokenizer

# Register custom model type
AutoConfig.register("char-engram", CharEngramConfig)
AutoModelForCausalLM.register(CharEngramConfig, CharLevelEngramModel)

# Load from local directory containing config + weights
model = AutoModelForCausalLM.from_pretrained("./char_engram_model")
tokenizer = ByteTokenizer()
```

---

## When to Use This Version

This byte‑level, tokenizer‑free Engram is useful when:

- You want Engram‑style static memory but **do not want to depend on any tokenizer**.
- You work with **mixed or noisy text** (logs, code, multilingual content) where sub‑word tokenizers are brittle.
- You want a **minimal, fully self‑contained Engram example** that:
  - can be trained end‑to‑end,
  - and can be integrated with existing Hugging Face infrastructure.

If you are already using the original Engram in a tokenized setup, you can treat this as an **alternative implementation** that demonstrates the same ideas (N‑gram memory + gated fusion) on a purely byte‑level architecture.

---

## Notes and Limitations

- **Hash collisions**: Different byte N‑grams may map to the same memory slot. Use a sufficiently large `memory_capacity` to reduce harmful collisions.
- **Byte‑level granularity**: All N‑grams are byte‑level, not semantic tokens. You may need:
  - longer N‑grams,
  - more training,
  - or auxiliary objectives if you want the memory to align with higher‑level concepts.
- **Simple tokenizer**: `ByteTokenizer` is intentionally minimal—no special tokens beyond using byte `0` as padding by default.

---








<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/"><img alt="Homepage"
    src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true"/></a>
  <a href="https://chat.deepseek.com/"><img alt="Chat"
    src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white"/></a>
  <a href="https://huggingface.co/deepseek-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/Tc7c45Zzu5"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true"><img alt="Wechat"
    src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://twitter.com/deepseek_ai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <br>
</div>

## 1. Introduction

This repository contains the official implementation for the paper: **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](Engram_paper.pdf)**.

> **Abstract:** While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup. To address this, we explore **conditional memory** as a complementary sparsity axis, instantiated via **Engram**, a module that modernizes classic $N$-gram embeddings for $\mathcal{O}(1)$ lookup.

**Key Contributions:**
- **Sparsity Allocation:** We formulate the trade-off between neural computation (MoE) and static memory (Engram), identifying a U-shaped scaling law that guides optimal capacity allocation.
- **Empirical Verification:** Under strict iso-parameter and iso-FLOPs constraints, the Engram-27B model demonstrates consistent improvements over MoE baselines across knowledge, reasoning, code and math domains.
- **Mechanistic Analysis:** Our analysis suggests that Engram relieves early layers from static pattern reconstruction, potentially preserving effective depth for complex reasoning.
- **System Efficiency:** The module employs deterministic addressing, enabling the offloading of massive embedding tables to host memory with minimal inference overhead.


## 2. Architecture

The Engram module augments the backbone by retrieving static $N$-gram memory and fusing it with dynamic hidden states. The architecture is shown below ([drawio provided](drawio/Engram.drawio)):

<p align="center">
  <img width="75%" src="figures/arch.png" alt="Engram Architecture">
</p>

## 3. Evaluation

### Scaling Law
<p align="center">
  <img width="90%" src="figures/scaling_law.png" alt="Scaling Law">
</p>

---

### Large Scale Pre-training
<p align="center">
  <img width="80%" src="figures/27b_exp_results.png" alt="Pre-training Results">
</p>

---

### Long-context Training
<p align="center">
  <img width="80%" src="figures/long_context_results.png" alt="Long Context Results">
</p>


## 4. Case Study of Engram
<p align="center">
  <img width="80%" src="figures/case.png" alt="Long Context Results">
</p>

## 5. Quick Start

We recommend using Python 3.8+ and PyTorch.
```bash
pip install torch numpy transformers sympy
```
We provide a standalone implementation to demonstrate the core logic of the Engram module:
```bash
python engram_demo_v1.py
```

> ⚠️ **Note:** The provided code is a demonstration version intended to illustrate the data flow. It mocks standard components (like Attention/MoE/mHC) to focus on the Engram module. 


## 6. License
The use of Engram models is subject to [the Model License](LICENSE).

## 7. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).
