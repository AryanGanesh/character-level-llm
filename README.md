# LLM From Scratch

A character-level Language Model built from scratch in Python and PyTorch,
following Andrej Karpathy's "Let's Build GPT" as a learning backbone.

## What This Is
This is a student project aimed at understanding how Large Language Models
actually work under the hood — not just using them via an API, but building
the core components from zero. The long-term goal is to fine-tune this into
a small, fully local personal assistant that runs entirely on-device with
no data leaving the machine.

## What's Built So Far
- Character-level tokenizer with custom encode/decode functions
- PyTorch tensor pipeline with train/validation splitting
- Batch engine with random sampling (get_batch)
- Bigram Language Model base class
- Self-attention head with Query, Key, Value mechanism
- Causal masking for autoregressive generation
- Text generation function with temperature sampling
- Training loop with Adam optimizer

## Dataset
Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) —
a dataset of simple English children's stories. Clean, small vocabulary,
ideal for small models.

## Stack
- Python 3.12
- PyTorch
- Hugging Face Datasets

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch datasets
python tokenizer.py
```

## Status
Currently implementing the Transformer architecture — multi-head attention,
feed-forward layers, and residual connections. Final training will run on
a college GPU.

## Reference
Based on [Andrej Karpathy's "Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
