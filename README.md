# A Large Language Model (LLM)

- **Category:** AI Model
- **Language:** Python (Pure Python 3.15 Standard Library)
- **Original Tutorial:** [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

## Overview
A self-contained, zero-dependency GPT-style decoder-only Large Language Model built from first principles in Python. It implements Byte-Pair Encoding (BPE) tokenization, a reverse-mode scalar autograd engine (`Value`), multi-head causal self-attention, pre-LayerNorm transformer blocks with GELU feed-forward networks, AdamW optimization, and autoregressive decoding.

## Architecture & Key Algorithms
- **BPE Tokenizer (`BPETokenizer`):** Learns subword merge rules by iteratively merging the most frequent adjacent symbol pairs and supports `<|endoftext|>` special tokens.
- **Autograd Engine (`Value`):** Dynamic DAG topological backpropagation supporting `exp`, `log`, `tanh`, `gelu`, and arithmetic gradients.
- **GPT Transformer (`GPTModel`):** Token + learned positional embeddings, scaled dot-product multi-head causal self-attention ($QK^T / \sqrt{d_k}$), residual connections, LayerNorm, and output language-model head.
- **Training & Decoding (`AdamW`, `generate`):** Trains with next-token cross-entropy loss and generates text via greedy, temperature-scaled, and top-$k$ sampling.

## How to Run
```bash
python main.py
python main.py --epochs 20 --prompt "the dog " --max-tokens 8 --temperature 0.7 --top-k 4
```

### CLI Options
- `--epochs`: Training epochs (default `18`)
- `--prompt`: Seed text for generation (default `"the cat "`)
- `--max-tokens`: Tokens to generate (default `6`)
- `--temperature`: Sampling temperature; `0.0` runs greedy decoding
- `--top-k`: Top-k sampling cutoff (`0` disables cutoff)
- `--seed`: Random seed for initialization and sampling

### Unit Tests
```bash
python -m unittest discover -s tests -v
```
