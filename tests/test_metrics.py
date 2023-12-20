import unittest

import torch

from src.metrics import perplexity


class TestPerplexity(unittest.TestCase):
    def test_perplexity(self):
        logits = [
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
        ]
        labels = [
            [0, 0, 0],
            [1, 1, 1],
        ]
        perplexities = perplexity(torch.tensor(logits), torch.tensor(labels)).tolist()
        self.assertEqual(len(perplexities), 2)
        self.assertAlmostEqual(perplexities[0], 1.0)
        self.assertNotAlmostEqual(perplexities[1], 1.0)
