# LLM From Scratch

> Building a character-level Transformer Language Model from zero, in Python and PyTorch.  
> No APIs. No shortcuts. Every component written and understood from the ground up.

---

## What Is This?

This is a student project that builds a working Language Model from scratch, following Andrej Karpathy's "Let's Build GPT" as a learning backbone.

The goal is not just to run a model — but to understand every single component: how text becomes numbers, how attention works, how the model learns, and how it generates new text.

Long-term target: fine-tune this into a small, fully local personal assistant that runs entirely on-device. No data leaves the machine.

---

## Full Model Pipeline

```mermaid
graph TD
    A[Raw Text - TinyStories] --> B[Tokenizer]
    B --> C[Token Integers]
    C --> D[Embedding Table]
    D --> E[Transformer Block 1]
    E --> F[Transformer Block 2]
    F --> G[Transformer Block 3]
    G --> H[LM Head]
    H --> I[Logits]
    I -->|Training| J[Cross-Entropy Loss]
    I -->|Generation| K[Softmax + Sample]
    K --> L[Generated Text]

    style A fill:#FFE4B5
    style B fill:#87CEEB
    style D fill:#87CEEB
    style E fill:#90EE90
    style F fill:#90EE90
    style G fill:#90EE90
    style H fill:#87CEEB
    style J fill:#FFB6C1
    style L fill:#FFB6C1
```

---

## Transformer Block

```mermaid
graph TD
    A[Input x] --> B[Multi-Head Attention]
    A --> C{ }
    B --> C
    C -->|x = attention + x| D[After Residual 1]
    D --> E[Feed-Forward Network]
    D --> F{ }
    E --> F
    F -->|x = ffwd + x| G[Output x]

    style B fill:#90EE90
    style E fill:#87CEEB
    style C fill:#FFE4B5
    style F fill:#FFE4B5
```

---

## Self-Attention (Single Head)

```mermaid
graph TD
    A[Input x] --> B[Query Linear Layer]
    A --> C[Key Linear Layer]
    A --> D[Value Linear Layer]
    B --> E[Q @ Kᵀ = raw scores]
    C --> E
    E --> F[Scale by 1 divided by sqrt head_size]
    F --> G[Apply Causal Mask - block future tokens]
    G --> H[Softmax = attention weights]
    H --> I[weights @ V = output]
    D --> I

    style B fill:#87CEEB
    style C fill:#87CEEB
    style D fill:#87CEEB
    style G fill:#FFB6C1
    style H fill:#90EE90
    style I fill:#FFE4B5
```

---

## Multi-Head Attention

```mermaid
graph TD
    A[Input x - B T 32] --> B[Head 1 - size 8]
    A --> C[Head 2 - size 8]
    A --> D[Head 3 - size 8]
    A --> E[Head 4 - size 8]
    B --> F[Concatenate all heads - B T 32]
    C --> F
    D --> F
    E --> F
    F --> G[Projection Linear 32 to 32]
    G --> H[Output - B T 32]

    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
    style F fill:#FFE4B5
    style G fill:#87CEEB
```

---

## Feed-Forward Network

```mermaid
graph TD
    A[Input - n_embd 32] --> B[Linear 32 to 128 - expand 4x]
    B --> C[ReLU - negatives become zero]
    C --> D[Linear 128 to 32 - compress back]
    D --> E[Output - n_embd 32]

    style B fill:#87CEEB
    style C fill:#FFB6C1
    style D fill:#87CEEB
```

---

## Training Loop

```mermaid
graph TD
    A[Start Training] --> B[get_batch - random chunk of text]
    B --> C[Forward Pass through model]
    C --> D[Calculate Loss - how wrong is it?]
    D --> E[loss.backward - compute gradients]
    E --> F[optimizer.step - update weights]
    F --> G{10000 steps done?}
    G -->|No| B
    G -->|Yes| H[Generate Text]

    style A fill:#90EE90
    style D fill:#FFB6C1
    style E fill:#87CEEB
    style F fill:#FFE4B5
    style H fill:#90EE90
```

---

## Loss Progress

| Stage | What Was Added | Loss |
|---|---|---|
| Start | Random weights, no training | ~4.90 |
| Chapter 3 | Bigram model baseline | ~4.90 |
| Chapter 5 | Training loop + optimizer | ~2.30 |
| Chapter 6 | Single-head self-attention | ~2.25 |
| Chapter 7 | Multi-head attention (4 heads) | ~2.21 |
| Chapter 8 | Feed-forward layer | ~2.05 |
| Chapter 9 | 3× Transformer blocks + residuals | ~1.55 |

---

## Generated Output Progress

**Before training — pure noise:**
```
!pdL.6œXw¡Vx!!BE4E«V-©;0Fœq!R g T˜Fu
```

**After training, no attention:**
```
pasthupppean a wassiliemmar pog fay wis stond
```

**After multi-head attention:**
```
Onerday's d. tifubupon th cary upedel Whs al
```

**After 3× Transformer blocks:**
```
Once upon a time there was a fary a ine, Lily.
"What'lll strainghid.
```

---

## Project Structure

```
llm-from-scratch/
├── .venv/           virtual environment
├── tokenizer.py     main model file
└── README.md
```

---

## Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Language |
| PyTorch | Neural network engine |
| Hugging Face Datasets | TinyStories dataset |

---

## Dataset

**[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** — 2.1 million simple English children's stories. Clean, small vocabulary, perfect for small models. Runs fully locally.

---

## Setup

```powershell
git clone https://github.com/AryanGanesh/llm-from-scratch
cd llm-from-scratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch datasets
python tokenizer.py
```

---

## Current Hyperparameters

| Parameter | Value |
|---|---|
| n_embd | 32 |
| block_size | 32 |
| batch_size | 32 |
| num_heads | 4 |
| num_blocks | 3 |
| learning rate | 1e-3 |
| training steps | 10,000 |

---

## What's Left to Build

- [ ] Layer Normalisation
- [ ] Positional Embeddings
- [ ] Scale up hyperparameters

---

## Reference

Based on [Andrej Karpathy's "Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## Author

Built session by session as a student project — intermediate Python to full Transformer from scratch.
