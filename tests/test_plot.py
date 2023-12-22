import tempfile
import unittest
from pathlib import Path

from src.plot import plot_perplexity
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
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/perplexity.pdf"
            plot_perplexity(examples, path)
            self.assertTrue(Path(path).exists())
