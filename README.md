LLM From Scratch

Building a character-level Transformer Language Model from zero, in Python and PyTorch.
No APIs. No shortcuts. Every component written and understood from the ground up.

What Is This?

This is a student project that builds a working Language Model from scratch, following Andrej Karpathy’s “Let’s Build GPT” as a learning backbone.

The goal is not just to run a model — but to understand every component:

how text becomes numbers
how attention works
how training actually updates weights
how generation emerges from probabilities

Long-term goal: a small, fully local personal assistant that runs entirely on-device.

Full Pipeline (Updated)

graph TD
    A["📄 Raw Text\nTinyStories"] --> B["🔤 Tokenizer\nchar → integer"]
    B --> C["🔢 Token Integers\n[79, 23, 45, ...]"]
    C --> D["📦 get_batch\nrandom chunks"]
    D --> E["🗂️ Embedding Table\ntoken → 32-dim vector"]

    E --> BLOCK

    subgraph BLOCK ["🟩 Transformer Block × 3 (Pre-Norm)"]
        direction TB

        LN1["LayerNorm 1"] --> MHA["Multi-Head Attention\n4 heads × size 8"]
        MHA --> DO1["Dropout"]
        DO1 -->|"Residual"| R1["x = attn(x) + x"]

        R1 --> LN2["LayerNorm 2"]
        LN2 --> FF["Feed-Forward\n32 → 128 → ReLU → 32"]
        FF --> DO2["Dropout"]
        DO2 -->|"Residual"| R2["x = ffwd(x) + x"]
    end

    R2 --> LNF["Final LayerNorm"]
    LNF --> I["LM Head\nLinear → Logits"]

    I -->|"Training"| J["Cross-Entropy Loss\n~1.6–1.8"]
    J --> K["Backpropagation"]
    K --> L["AdamW Optimizer\n10,000 steps"]
    L --> D

    I -->|"Generation"| M["Softmax → Sample"]
    M --> O["📝 Generated Text"]

    style LN1 fill:#f9f,stroke:#333
    style LN2 fill:#f9f,stroke:#333
    style LNF fill:#f9f,stroke:#333
    style DO1 fill:#fab,stroke:#333
    style DO2 fill:#fab,stroke:#333
    style BLOCK fill:#f0f0f0,stroke:#333,stroke-dasharray: 5 5

This version reflects your Pre-Norm Transformer, with LayerNorm + Dropout fully integrated.

Loss Progress
Stage	What Was Added	Loss
Start	Random initialization	~4.90
Chapter 5	Training loop + AdamW	~2.30
Chapter 6–8	Attention + FFN	~2.05
Chapter 9	3× Blocks + Residuals	~1.55
Chapter 10	Layer Normalization	~1.52
Chapter 11	Dropout (regularization)	~1.64 – 1.80

Note: Slight loss increase after dropout is expected — generalization improves.

Generated Output Progress

Before training — noise

!pdL.6œXw¡Vx!!BE4E«V-©;0Fœq!R g T˜Fu

After training (no attention)

pasthupppean a wassiliemmar pog fay wis stond

After attention + blocks

Once upon a time there was a fary a ine, Lily.
"What'lll strainghid.
Project Structure
character-level-llm/
├── .venv/
├── tokenizer.py
├── requirements.txt
└── README.md
Stack
Tool	Purpose
Python 3.12	Core language
PyTorch	Neural network engine
Hugging Face Datasets	Dataset loading
Dataset

TinyStories — a dataset of ~2M short English stories designed for small language models.

Clean vocabulary
Short sequences
Ideal for fast iteration
Setup
git clone https://github.com/AryanGanesh/character-level-llm
cd character-level-llm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch datasets
python tokenizer.py
Current Hyperparameters
Parameter	Value
n_embd	32
block_size	32
batch_size	32
num_heads	4
num_blocks	3
dropout	0.1
learning rate	1e-3
training steps	10,000
architecture	Pre-Norm Transformer
Architecture Notes
Pre-Norm design → LayerNorm applied before attention and FFN
Residual connections → stable gradient flow
Dropout (0.1) → prevents memorization
Final LayerNorm → stabilizes logits before prediction

This matches modern Transformer implementations used in production models.

What's Left to Build
 Layer Normalization
 Dropout
 Positional Embeddings
 Scale up (n_embd = 384, n_layer = 6) on GPU
Key Insight (Current Stage)

Your model is now:

✅ Deep (3 Transformer blocks)
✅ Stable (LayerNorm + residuals)
✅ Regularized (Dropout)
❗ Still position-blind

Which leads directly to the next milestone:

👉 Positional Embeddings (Chapter 12)

Reference

Based on:
Andrej Karpathy — Let’s Build GPT

Author

AryanGanesh
