import unittest

import torch

from src.metrics import extractable, min_k_percent_prob, perplexity


class TestPerplexity(unittest.TestCase):
    def test_perplexity(self):
        logits = [
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
        ]
        labels = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
        ]
        perplexities = perplexity(torch.tensor(logits), torch.tensor(labels)).tolist()
        self.assertEqual(len(perplexities), 3)
        self.assertAlmostEqual(perplexities[0], 1.0)
        self.assertGreater(perplexities[1], 1.0)
        self.assertGreater(perplexities[2], 1.0)


class TestMinKPercentProb(unittest.TestCase):
    def test_min_k_percent_prob(self):
        logits = [
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
            [
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
                [100.0, -100.0],
            ],
        ]
        labels = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
        ]
        min_k_percent_prob_ = min_k_percent_prob(torch.tensor(logits), torch.tensor(labels), k=20.0).tolist()
        self.assertEqual(len(min_k_percent_prob_), 3)
        self.assertAlmostEqual(min_k_percent_prob_[0], 0.0)
        self.assertLess(min_k_percent_prob_[1], 0.0)
        self.assertLess(min_k_percent_prob_[2], 0.0)


class TestExtractable(unittest.TestCase):
    def test_extractable(self):
        output_ids = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
        labels = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
        ]
        extractables = extractable(torch.tensor(output_ids), torch.tensor(labels)).tolist()
        self.assertEqual(len(extractables), 3)
        self.assertAlmostEqual(extractables[0], 1.0)
        self.assertAlmostEqual(extractables[1], 1.0)
        self.assertAlmostEqual(extractables[2], 0.0)
