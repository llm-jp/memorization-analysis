import unittest

from src.plot import plot_verbatim_memorization_ratio
from src.utils import Example


class TestPlotVerbatimMemorizationRatio(unittest.TestCase):
    def test_plot_verbatim_memorization_ratio(self):
        examples = [
            Example(
                iteration=0,
                dataset_idx=0,
                dataset_name="test",
                doc_ids=[0],
                text="test",
                token_ids=[0],
                metrics={
                    "extractable/100": 2.0,
                    "extractable/200": 12.0,
                    "extractable/500": 22.0,
                    "extractable/1000": 32.0,
                },
                completion_stats={
                    "last_iteration": 0,
                    "count": 1,
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
                    "extractable/100": 2.0,
                    "extractable/200": 12.0,
                    "extractable/500": 22.0,
                    "extractable/1000": 32.0,
                },
                completion_stats={
                    "last_iteration": 0,
                    "count": 1,
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
                    "extractable/100": 2.0,
                    "extractable/200": 12.0,
                    "extractable/500": 22.0,
                    "extractable/1000": 32.0,
                },
                completion_stats={
                    "last_iteration": 1,
                    "count": 1,
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
                    "extractable/100": 2.0,
                    "extractable/200": 12.0,
                    "extractable/500": 22.0,
                    "extractable/1000": 32.0,
                },
                completion_stats={
                    "last_iteration": 1,
                    "count": 1,
                },
            ),
        ]
        _ = plot_verbatim_memorization_ratio(examples)
