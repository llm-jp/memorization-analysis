import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.evaluate import logits


class TestLogits(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50688  # keeping the same vocab size as llm-jp-1.3b-v1.0
        config = GPT2Config(
            activation_function="gelu",
            n_positions=2048,
            n_ctx=1024,
            n_embd=2,  # smaller embedding size
            n_head=2,  # fewer attention heads
            n_layer=2,  # fewer layers
            vocab_size=self.vocab_size,
        )
        self.model = GPT2LMHeadModel(config)

    def test_logits(self):
        input_ids = [
            [0, 0, 0],
            [0, 0, 0],
        ]
        logits_ = logits(self.model, torch.tensor(input_ids))
        self.assertEqual(logits_.size(), (2, 3, self.vocab_size))
