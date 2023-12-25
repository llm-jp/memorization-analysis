import unittest

from src.plot import plot_extractable, plot_min_k_percent_prob, plot_perplexity
from src.utils import Example


class TestPlotPerplexity(unittest.TestCase):
    def test_plot_perplexity(self):
        examples = [
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={"perplexity": 1.0},
            ),
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={"perplexity": 2.0},
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={"perplexity": 2.0},
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={"perplexity": 3.0},
            ),
        ]
        _ = plot_perplexity(examples)


class TestPlotMinKPercentProb(unittest.TestCase):
    def test_plot_min_k_percent_prob(self):
        examples = [
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "min_k_percent_prob/20": 1.0,
                    "min_k_percent_prob/30": 11.0,
                },
            ),
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "min_k_percent_prob/20": 2.0,
                    "min_k_percent_prob/30": 12.0,
                },
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "min_k_percent_prob/20": 2.0,
                    "min_k_percent_prob/30": 12.0,
                },
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "min_k_percent_prob/20": 3.0,
                    "min_k_percent_prob/30": 13.0,
                },
            ),
        ]
        _ = plot_min_k_percent_prob(examples)


class TestPlotExtractable(unittest.TestCase):
    def test_plot_extractable(self):
        examples = [
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "extractable/20": 1.0,
                    "extractable/30": 11.0,
                },
            ),
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "extractable/20": 2.0,
                    "extractable/30": 12.0,
                },
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "extractable/20": 2.0,
                    "extractable/30": 12.0,
                },
            ),
            Example(
                iteration=1,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "extractable/20": 3.0,
                    "extractable/30": 13.0,
                },
            ),
        ]
        _ = plot_extractable(examples)
