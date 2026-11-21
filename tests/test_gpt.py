"""Unit tests for BPE tokenizer, Value scalar autograd, causal masking, and GPT generation."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import BPETokenizer, GPTModel, Value  # noqa: E402


def finite_diff(fn, x_val: float, eps: float = 1e-6) -> float:
    return (fn(Value(x_val + eps)).data - fn(Value(x_val - eps)).data) / (2.0 * eps)


class TestBPETokenizer(unittest.TestCase):
    def test_roundtrip_and_special_token(self) -> None:
        corpus = (
            "the cat sat on the mat <|endoftext|> "
            "the dog sat near the log <|endoftext|> "
            "the cat ate fish"
        )
        tok = BPETokenizer()
        tok.train(corpus, num_merges=20)
        sample = "the cat sat on the mat <|endoftext|>"
        ids = tok.encode(sample)
        self.assertGreater(len(tok.merges), 0)
        self.assertIn(tok.inverse_vocab["<|endoftext|>"], ids)
        self.assertEqual(tok.decode(ids), sample)


class TestAutogradValue(unittest.TestCase):
    def test_finite_difference_gradients(self) -> None:
        def f(x: Value) -> Value:
            y = Value(1.25)
            return ((x * y + (x - 0.3) ** 2).tanh() + (x * 0.5).gelu() + (x * 0.2).exp() + (x + 2.0).log())

        x_val = 0.75
        x = Value(x_val)
        out = f(x)
        out.backward()
        expected = finite_diff(f, x_val)
        self.assertAlmostEqual(x.grad, expected, places=5)

    def test_sub_and_div_gradients(self) -> None:
        a = Value(3.0)
        b = Value(1.5)
        out = (a - b) / (a + b)
        out.backward()

        da = ((3.0 + 1e-6 - 1.5) / (3.0 + 1e-6 + 1.5) - (3.0 - 1e-6 - 1.5) / (3.0 - 1e-6 + 1.5)) / 2e-6
        db = ((3.0 - (1.5 + 1e-6)) / (3.0 + 1.5 + 1e-6) - (3.0 - (1.5 - 1e-6)) / (3.0 + 1.5 - 1e-6)) / 2e-6
        self.assertAlmostEqual(a.grad, da, places=5)
        self.assertAlmostEqual(b.grad, db, places=5)


class TestGPTModel(unittest.TestCase):
    def test_causal_mask_invariance(self) -> None:
        model = GPTModel(vocab_size=12, ctx_len=8, d_model=8, n_heads=2, n_layers=1, seed=11)
        seq_a = [1, 4, 2, 7]
        seq_b = [1, 4, 2, 9]

        logits_a = model.forward(seq_a)
        logits_b = model.forward(seq_b)

        # Earlier positions (t = 0, 1, 2) must not depend on token 3
        for t in range(3):
            for va, vb in zip(logits_a[t], logits_b[t]):
                self.assertAlmostEqual(va.data, vb.data, places=10)

    def test_greedy_generation_is_deterministic_and_grows(self) -> None:
        model = GPTModel(vocab_size=10, ctx_len=8, d_model=8, n_heads=2, n_layers=1, seed=5)
        prompt = [1, 2, 3]
        out1 = model.generate(prompt, max_new_tokens=4, temperature=0.0)
        out2 = model.generate(prompt, max_new_tokens=4, temperature=0.0)
        self.assertEqual(out1, out2)
        self.assertEqual(len(out1), len(prompt) + 4)
        self.assertEqual(out1[:3], prompt)


if __name__ == "__main__":
    unittest.main()
