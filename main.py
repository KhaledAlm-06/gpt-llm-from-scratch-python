"""A Large Language Model (LLM) from Scratch in pure Python 3.15 stdlib.

Implements Sebastian Raschka's 'LLMs from scratch' core stack:
1. Byte-Pair Encoding (BPE) Tokenizer trained from corpus + encode/decode + <|endoftext|>
2. Autograd/Manual-Backprop Tensor Matrix Ops (LayerNorm, GELU, Causal Self-Attention, Softmax, Cross-Entropy)
3. GPT-2 Architecture (Token + Positional Embeddings, Multi-Head Causal Attention, Pre-LN Transformer Blocks, LM Head)
4. AdamW training loop demonstrating loss convergence on a multi-sentence corpus
5. Autoregressive text generation with greedy, temperature, and top-k sampling.
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ========================= 1. BPE TOKENIZER =========================


class BPETokenizer:
    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.vocab: Dict[int, str] = {}
        self.inverse_vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []

    def train(self, text: str, num_merges: int = 32) -> None:
        chars = sorted(set(text))
        self.vocab = {}
        for idx, tok in enumerate(self.special_tokens):
            self.vocab[idx] = tok
        offset = len(self.vocab)
        for i, ch in enumerate(chars):
            self.vocab[offset + i] = ch
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Split corpus into whitespace-preserving pre-tokens
        words = re.findall(r"\S+\s*", text)
        word_seqs: List[List[str]] = [list(w) for w in words]

        self.merges = []
        for _ in range(num_merges):
            pairs: Counter[Tuple[str, str]] = Counter()
            for seq in word_seqs:
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += 1
            if not pairs:
                break
            best_pair, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if merged not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = merged
                self.inverse_vocab[merged] = new_id

            new_seqs: List[List[str]] = []
            for seq in word_seqs:
                i = 0
                out: List[str] = []
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                new_seqs.append(out)
            word_seqs = new_seqs

    def encode(self, text: str) -> List[int]:
        special_pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(special_pattern, text)
        ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.inverse_vocab and part in self.special_tokens:
                ids.append(self.inverse_vocab[part])
                continue
            for word in re.findall(r"\S+\s*", part):
                seq = list(word)
                for a, b in self.merges:
                    i = 0
                    updated: List[str] = []
                    while i < len(seq):
                        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                            updated.append(a + b)
                            i += 2
                        else:
                            updated.append(seq[i])
                            i += 1
                    seq = updated
                for tok in seq:
                    if tok in self.inverse_vocab:
                        ids.append(self.inverse_vocab[tok])
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.vocab[i] for i in ids)


# ========================= 2. REVERSE-MODE AUTOGRAD ENGINE =========================


class Value:
    """Scalar autograd node building a dynamic computation graph."""

    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data: float, _children: Tuple["Value", ...] = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other: object) -> "Value":
        o = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data + o.data, (self, o))

        def _backward() -> None:
            self.grad += out.grad
            o.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: object) -> "Value":
        return self + other

    def __sub__(self, other: object) -> "Value":
        o = other if isinstance(other, Value) else Value(float(other))
        return self + (-o)

    def __neg__(self) -> "Value":
        return self * -1.0

    def __mul__(self, other: object) -> "Value":
        o = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data * o.data, (self, o))

        def _backward() -> None:
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: object) -> "Value":
        return self * other

    def __truediv__(self, other: object) -> "Value":
        o = other if isinstance(other, Value) else Value(float(other))
        return self * (o ** -1.0)

    def __pow__(self, exponent: float) -> "Value":
        out = Value(self.data ** exponent, (self,))

        def _backward() -> None:
            self.grad += (exponent * (self.data ** (exponent - 1.0))) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(min(60.0, max(-60.0, self.data)))
        out = Value(e, (self,))

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Value":
        c = math.sqrt(2.0 / math.pi)
        inner = c * (self.data + 0.044715 * (self.data ** 3))
        th = math.tanh(inner)
        val = 0.5 * self.data * (1.0 + th)
        out = Value(val, (self,))

        def _backward() -> None:
            sech2 = 1.0 - th * th
            d_inner = c * (1.0 + 3.0 * 0.044715 * self.data * self.data)
            d_gelu = 0.5 * (1.0 + th) + 0.5 * self.data * sech2 * d_inner
            self.grad += d_gelu * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        val = math.log(max(self.data, 1e-12))
        out = Value(val, (self,))

        def _backward() -> None:
            self.grad += (1.0 / max(self.data, 1e-12)) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: List[Value] = []
        visited = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def linear(x: List[Value], weight: List[List[Value]], bias: Optional[List[Value]] = None) -> List[Value]:
    out: List[Value] = []
    for row_idx, w_row in enumerate(weight):
        acc = sum((w_row[j] * x[j] for j in range(len(x))), Value(0.0))
        if bias is not None:
            acc = acc + bias[row_idx]
        out.append(acc)
    return out


def layer_norm(x: List[Value], gamma: List[Value], beta: List[Value], eps: float = 1e-5) -> List[Value]:
    n = len(x)
    mean = sum(x, Value(0.0)) * (1.0 / n)
    var = sum(((xi - mean) ** 2 for xi in x), Value(0.0)) * (1.0 / n)
    inv_std = (var + eps) ** -0.5
    return [gamma[i] * ((x[i] - mean) * inv_std) + beta[i] for i in range(n)]


def softmax_values(logits: List[Value]) -> List[Value]:
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    denom = sum(exps, Value(0.0))
    return [e / denom for e in exps]


# ========================= 3. GPT ARCHITECTURE =========================


class GPTModel:
    def __init__(
        self,
        vocab_size: int,
        ctx_len: int = 16,
        d_model: int = 16,
        n_heads: int = 2,
        n_layers: int = 1,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers

        def rand_mat(rows: int, cols: int, std: float = 0.12) -> List[List[Value]]:
            return [[Value(rng.gauss(0.0, std)) for _ in range(cols)] for _ in range(rows)]

        def ones(n: int) -> List[Value]:
            return [Value(1.0) for _ in range(n)]

        def zeros(n: int) -> List[Value]:
            return [Value(0.0) for _ in range(n)]

        self.tok_emb = rand_mat(vocab_size, d_model)
        self.pos_emb = rand_mat(ctx_len, d_model)

        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                "ln1_g": ones(d_model),
                "ln1_b": zeros(d_model),
                "wq": rand_mat(d_model, d_model),
                "wk": rand_mat(d_model, d_model),
                "wv": rand_mat(d_model, d_model),
                "wo": rand_mat(d_model, d_model),
                "ln2_g": ones(d_model),
                "ln2_b": zeros(d_model),
                "fc1": rand_mat(d_model * 2, d_model),
                "fc1_b": zeros(d_model * 2),
                "fc2": rand_mat(d_model, d_model * 2),
                "fc2_b": zeros(d_model),
            })

        self.ln_f_g = ones(d_model)
        self.ln_f_b = zeros(d_model)
        self.lm_head = rand_mat(vocab_size, d_model)

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for row in self.tok_emb + self.pos_emb + self.lm_head:
            params.extend(row)
        params.extend(self.ln_f_g)
        params.extend(self.ln_f_b)
        for blk in self.blocks:
            for k, v in blk.items():
                if isinstance(v[0], list):
                    for r in v:
                        params.extend(r)
                else:
                    params.extend(v)
        return params

    def forward(self, token_ids: List[int]) -> List[List[Value]]:
        T = len(token_ids)
        assert T <= self.ctx_len
        x = [
            [self.tok_emb[token_ids[t]][d] + self.pos_emb[t][d] for d in range(self.d_model)]
            for t in range(T)
        ]

        scale = 1.0 / math.sqrt(self.head_dim)
        for blk in self.blocks:
            # Pre-LN + Multi-Head Causal Self-Attention
            normed1 = [layer_norm(xt, blk["ln1_g"], blk["ln1_b"]) for xt in x]
            Q = [linear(nt, blk["wq"]) for nt in normed1]
            K = [linear(nt, blk["wk"]) for nt in normed1]
            V = [linear(nt, blk["wv"]) for nt in normed1]

            attn_concat: List[List[Value]] = [[] for _ in range(T)]
            for h in range(self.n_heads):
                hs = h * self.head_dim
                he = hs + self.head_dim
                for t in range(T):
                    # Causal attention: only attend to s <= t
                    scores = []
                    for s in range(t + 1):
                        dot = sum((Q[t][hs + d] * K[s][hs + d] for d in range(self.head_dim)), Value(0.0))
                        scores.append(dot * scale)
                    weights = softmax_values(scores)
                    for d in range(self.head_dim):
                        head_out = sum((weights[s] * V[s][hs + d] for s in range(t + 1)), Value(0.0))
                        attn_concat[t].append(head_out)

            attn_proj = [linear(ac, blk["wo"]) for ac in attn_concat]
            x = [[x[t][d] + attn_proj[t][d] for d in range(self.d_model)] for t in range(T)]

            # Pre-LN + FeedForward (GELU)
            normed2 = [layer_norm(xt, blk["ln2_g"], blk["ln2_b"]) for xt in x]
            ff_out = []
            for nt in normed2:
                hidden = [v.gelu() for v in linear(nt, blk["fc1"], blk["fc1_b"])]
                ff_out.append(linear(hidden, blk["fc2"], blk["fc2_b"]))
            x = [[x[t][d] + ff_out[t][d] for d in range(self.d_model)] for t in range(T)]

        final_norm = [layer_norm(xt, self.ln_f_g, self.ln_f_b) for xt in x]
        return [linear(ft, self.lm_head) for ft in final_norm]

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 12,
        temperature: float = 0.7,
        top_k: int = 4,
        seed: int = 7,
    ) -> List[int]:
        rng = random.Random(seed)
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = ids[-self.ctx_len :]
            logits = [v.data for v in self.forward(ctx)[-1]]
            if temperature <= 0.0:
                next_id = max(range(len(logits)), key=lambda i: logits[i])
            else:
                scaled = [l / temperature for l in logits]
                ranked = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
                max_s = scaled[ranked[0]]
                probs = [math.exp(scaled[i] - max_s) for i in ranked]
                total = sum(probs)
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                next_id = ranked[-1]
                for idx, p in zip(ranked, probs):
                    cum += p
                    if r <= cum:
                        next_id = idx
                        break
            ids.append(next_id)
        return ids


class AdamW:
    def __init__(self, params: List[Value], lr: float = 0.05, weight_decay: float = 0.01) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0

    def step(self) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, p in enumerate(self.params):
            g = max(-5.0, min(5.0, p.grad))
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)
            p.data -= self.lr * (m_hat / (math.sqrt(v_hat) + eps) + self.weight_decay * p.data)


def main() -> None:
    corpus = (
        "the cat sat on the mat <|endoftext|> "
        "the cat ate the fish <|endoftext|> "
        "the dog sat on the rug <|endoftext|> "
    )

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=16)
    sample_text = "the cat sat on the mat <|endoftext|>"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text, f"Roundtrip mismatch: {decoded!r}"
    assert tokenizer.inverse_vocab["<|endoftext|>"] in encoded

    tokens = tokenizer.encode(corpus)
    model = GPTModel(vocab_size=len(tokenizer.vocab), ctx_len=16, d_model=8, n_heads=2, n_layers=1, seed=42)
    optimizer = AdamW(model.parameters(), lr=0.08, weight_decay=0.005)

    # Train on sliding sequence windows
    windows = [tokens[i : i + 7] for i in range(0, min(len(tokens) - 7, 8), 4)]
    losses: List[float] = []
    for epoch in range(18):
        epoch_loss = 0.0
        for win in windows:
            x_seq = win[:-1]
            y_seq = win[1:]
            optimizer.zero_grad()
            logits_seq = model.forward(x_seq)
            loss = Value(0.0)
            for logits, target in zip(logits_seq, y_seq):
                probs = softmax_values(logits)
                loss = loss + (-probs[target].log())
            loss = loss * (1.0 / len(y_seq))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
        losses.append(epoch_loss / len(windows))

    assert losses[-1] < losses[0] * 0.55, f"Expected training loss to drop sharply: {losses[0]:.3f} -> {losses[-1]:.3f}"

    prompt = tokenizer.encode("the cat ")
    gen_ids = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    gen_text = tokenizer.decode(gen_ids)
    assert gen_text.startswith("the cat ") and len(gen_text) > len("the cat ")

    print("=== A Large Language Model (GPT-2 + BPE + Autograd) Passed ===")
    print(f"BPE vocab size: {len(tokenizer.vocab)} | Merges learned: {len(tokenizer.merges)}")
    print(f"GPT parameters: {len(model.parameters())}")
    print(f"Training cross-entropy loss: epoch 1 = {losses[0]:.4f} -> epoch {len(losses)} = {losses[-1]:.4f}")
    print(f"Greedy generation from 'the cat ': {gen_text!r}")


if __name__ == "__main__":
    main()
