import unittest

from src.utils import Example


class TestExample(unittest.TestCase):
    def test_example(self):
        example = Example(
            iteration=0,
            dataset_idx=0,
            dataset_name="test",
            doc_ids=[0],
            text="test",
            token_ids=[0],
        )
        self.assertEqual(example.iteration, 0)
        self.assertEqual(example.dataset_idx, 0)
        self.assertEqual(example.dataset_name, "test")
        self.assertEqual(example.doc_ids, [0])
        self.assertEqual(example.text, "test")
        self.assertEqual(example.token_ids, [0])

    def test_example_from_json(self):
        json_obj = {
            "iteration": 0,
            "dataset_idx": 0,
            "dataset_name": "test",
            "doc_ids": [0],
            "text": "test",
            "token_ids": [0],
        }
        example = Example.from_json(json_obj)
        self.assertEqual(example.iteration, 0)
        self.assertEqual(example.dataset_idx, 0)
        self.assertEqual(example.dataset_name, "test")
        self.assertEqual(example.doc_ids, [0])
        self.assertEqual(example.text, "test")
        self.assertEqual(example.token_ids, [0])
