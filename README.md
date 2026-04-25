#  LLM From Scratch

> Building a character-level Transformer Language Model from zero, in Python and PyTorch.  
> No APIs. No shortcuts. Every component written and understood from the ground up.

---

## What Is This?

This is a student project that builds a working Language Model from scratch, following Andrej Karpathy's "Let's Build GPT" as a learning backbone.

The goal is not just to run a model — but to understand every single component: how text becomes numbers, how attention works, how the model learns, and how it generates new text.

Long-term target: fine-tune this into a small, fully local personal assistant that runs entirely on-device. No data leaves the machine.

---

## Full Pipeline

```mermaid
graph TD
    A["📄 Raw Text\nTinyStories"] --> B["🔤 Tokenizer\nchar → integer\nencode / decode"]
    B --> C["🔢 Token Integers\n[79, 23, 45, 12, ...]"]
    C --> D["📦 get_batch\nrandom chunks → x and y\nshifted by 1 position"]
    D --> E["🗂️ Embedding Table\ntoken integer → 32-dim vector"]

    E --> F

    subgraph BLOCK ["🟩 Transformer Block × 3  —  Pre-Norm architecture"]
        F["LayerNorm → Multi-Head Attention\n4 heads × size 8\nQ @ Kᵀ · scale · mask · softmax · dropout · V\n+ projection dropout"] -->|"x = attn(ln1(x)) + x  residual"| G["LayerNorm → Feed-Forward\n32 → 128 → ReLU → 32"]
        G -->|"x = ffwd(ln2(x)) + x  residual"| H["Output x"]
    end

    H --> I["Final LayerNorm"]
    I --> J["LM Head\nLinear → Logits\n[B, T, vocab_size]"]

    J -->|"Training"| K["Cross-Entropy Loss\ncompare logits vs y\n4.90 → 1.55"]
    K --> L["loss.backward\ngradients through every weight"]
    L --> M["AdamW optimizer.step\nnudge weights × 10,000"]
    M -->|"next batch"| D

    J -->|"Generation"| N["Softmax → Probabilities"]
    N --> O["torch.multinomial\nsample 1 token"]
    O -->|"append → loop"| P["📝 Generated Text\nOnce upon a time..."]

    style A fill:#FFE4B5,color:#000
    style B fill:#87CEEB,color:#000
    style C fill:#87CEEB,color:#000
    style D fill:#87CEEB,color:#000
    style E fill:#87CEEB,color:#000
    style F fill:#90EE90,color:#000
    style G fill:#87CEEB,color:#000
    style H fill:#90EE90,color:#000
    style I fill:#DDA0DD,color:#000
    style J fill:#87CEEB,color:#000
    style K fill:#FFB6C1,color:#000
    style L fill:#FFB6C1,color:#000
    style M fill:#FFE4B5,color:#000
    style N fill:#90EE90,color:#000
    style O fill:#90EE90,color:#000
    style P fill:#FFB6C1,color:#000
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
| Chapter 10 | Layer norm + dropout | ~1.55 → more stable |

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

## Architecture Details

### Pre-Norm Transformer Block
Each block applies Layer Normalisation *before* the sub-layer, not after. This is the modern convention (used in GPT-2 onwards) and stabilises training in deeper networks.

```
x = MultiHeadAttention(LayerNorm(x)) + x
x = FeedForward(LayerNorm(x)) + x
```

A final `LayerNorm` is applied after all three blocks before the LM head.

### Dropout
Dropout (p=0.1) is applied in two places per block:
- After the softmax attention weights inside each `Head`, before multiplying by Values
- After the projection layer in `MultiHeadAttention`

This randomly zeroes activations during training, preventing the model from over-relying on any single path and reducing overfitting.

---

## Project Structure

```
character-level-llm/
├── .venv/           virtual environment
├── tokenizer.py     main model file
├── requirements.txt
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
git clone https://github.com/AryanGanesh/character-level-llm
cd character-level-llm
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

## What's Left to Build

- [x] Layer Normalisation
- [x] Dropout
- [ ] Positional Embeddings
- [ ] Scale up hyperparameters
- [ ] Full training run on college GPU
- [ ] Fine-tune into personal assistant

---

## Reference

Based on [Andrej Karpathy's "Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## Author

AryanGanesh Kavuri — built session by session as a student project, intermediate Python to full Transformer from scratch.
