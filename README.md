#  LLM From Scratch

> Building a character-level Transformer Language Model from zero, in Python and PyTorch.
> No APIs. No shortcuts. Every component written and understood from the ground up.

---

## What Is This?

A student project that builds a working Language Model from scratch, following Andrej Karpathy's "Let's Build GPT" as a learning backbone.

The goal is not just to run a model — but to understand every single component: how text becomes numbers, how attention works, how the model learns, and how it generates new text.

Long-term target: fine-tune this into a small, fully local personal assistant that runs entirely on-device. No data leaves the machine.

---

## Full Model Pipeline

```mermaid
graph TD
    A[Raw Text - TinyStories] --> B[Tokenizer]
    B --> C[Token Integers]
    C --> D[Token Embedding Table]
    C --> E[Position Embedding Table]
    D --> F[Add Together]
    E --> F
    F --> G[Transformer Block 1]
    G --> H[Transformer Block 2]
    H --> I[Transformer Block 3]
    I --> J[Layer Norm]
    J --> K[LM Head]
    K --> L[Logits]
    L -->|Training| M[Cross-Entropy Loss]
    L -->|Generation| N[Softmax + Sample]
    N --> O[Generated Text]

    style A fill:#FFE4B5
    style B fill:#87CEEB
    style D fill:#87CEEB
    style E fill:#87CEEB
    style F fill:#FFE4B5
    style G fill:#90EE90
    style H fill:#90EE90
    style I fill:#90EE90
    style J fill:#DDA0DD
    style M fill:#FFB6C1
    style O fill:#FFB6C1
```

---

## Transformer Block

```mermaid
graph TD
    A[Input x] --> B[Layer Norm 1]
    B --> C[Multi-Head Attention]
    C --> D[Add Residual]
    A --> D
    D --> E[Layer Norm 2]
    E --> F[Feed-Forward Network]
    F --> G[Add Residual]
    D --> G
    G --> H[Output x]

    style B fill:#DDA0DD
    style C fill:#90EE90
    style D fill:#FFE4B5
    style E fill:#DDA0DD
    style F fill:#87CEEB
    style G fill:#FFE4B5
```

---

## Self-Attention Head

```mermaid
graph TD
    A[Input x] --> B[Query Linear]
    A --> C[Key Linear]
    A --> D[Value Linear]
    B --> E[Q times K-transpose]
    C --> E
    E --> F[Scale by 1 over sqrt head_size]
    F --> G[Causal Mask - block future tokens]
    G --> H[Softmax]
    H --> I[Dropout]
    I --> J[Multiply by V]
    D --> J
    J --> K[Output]

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
    A[Input x] --> B[Head 1]
    A --> C[Head 2]
    A --> D[Head 3]
    A --> E[Head 4]
    B --> F[Concatenate]
    C --> F
    D --> F
    E --> F
    F --> G[Projection Linear]
    G --> H[Dropout]
    H --> I[Output]

    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
    style F fill:#FFE4B5
    style G fill:#87CEEB
    style H fill:#FFE4B5
```

---

## Training Loop

```mermaid
graph TD
    A[Start] --> B[get_batch - random chunk]
    B --> C[Forward Pass]
    C --> D[Calculate Loss]
    D --> E[zero_grad]
    E --> F[loss.backward]
    F --> G[optimizer.step]
    G --> H{10000 steps done?}
    H -->|No| B
    H -->|Yes| I[Generate Text]

    style A fill:#90EE90
    style D fill:#FFB6C1
    style F fill:#87CEEB
    style G fill:#FFE4B5
    style I fill:#90EE90
```

---

## Loss Progress

| Stage | What Was Added | Loss |
|---|---|---|
| Start | Random weights, no training | ~4.90 |
| Chapter 3 | Bigram model + training | ~2.30 |
| Chapter 6 | Single-head self-attention | ~2.25 |
| Chapter 7 | Multi-head attention (4 heads) | ~2.21 |
| Chapter 8 | Feed-forward layer | ~2.05 |
| Chapter 9 | 3× Transformer blocks + residuals | ~1.55 |
| Chapter 10 | Layer normalisation | ~1.52 |
| Chapter 11 | Dropout | ~1.60–1.80 |
| Chapter 12 | Positional embeddings | ~1.54 |

---

## Generated Output Progress

**Before training:**
```
!pdL.6œXw¡Vx!!BE4E«V-©;0Fœq
```

**After Bigram + training:**
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
```

**After positional embeddings:**
```
One day, supriestled exploking to gets!
"Whould his reags day walls."
Tim you. Thad a awant hering yodded said,
"Timmy, santild youthere eactays was bles."
```

---

## Architecture Built

| Component | Purpose |
|---|---|
| Tokenizer | Characters → integers and back |
| Token Embeddings | Integer → meaningful vector |
| Positional Embeddings | Position → vector, added to token embedding |
| Multi-Head Attention | Tokens communicate — 4 heads in parallel |
| Feed-Forward Network | Each token thinks independently |
| Residual Connections | Gradient highway through deep network |
| Layer Normalisation | Keeps values stable through deep layers |
| Dropout | Prevents memorisation, improves generalisation |

---

## Project Structure

```
llm-from-scratch/
├── .venv/           virtual environment
├── tokenizer.py     full model
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

**[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** — 2.1 million simple English children's stories. Clean, small vocabulary, perfect for small models. Runs fully locally — no data sent anywhere.

---

## Setup

```powershell
git clone https://github.com/yourusername/llm-from-scratch
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
| dropout | 0.1 |
| learning rate | 1e-3 |
| training steps | 10,000 |

---

## What's Next

- [ ] Scale up hyperparameters
- [ ] Full training run on college GPU
- [ ] Fine-tune into personal assistant

---

## Reference

Based on [Andrej Karpathy's "Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## Author
AryanGanesh- built session by session as a student project
