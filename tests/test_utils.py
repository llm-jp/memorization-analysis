import dataclasses
import gzip
import json
import tempfile
import unittest

from src.utils import Example, load_file


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
        example = Example(**json_obj)
        self.assertEqual(example.iteration, 0)
        self.assertEqual(example.dataset_idx, 0)
        self.assertEqual(example.dataset_name, "test")
        self.assertEqual(example.doc_ids, [0])
        self.assertEqual(example.text, "test")
        self.assertEqual(example.token_ids, [0])


class TestLoadFile(unittest.TestCase):
    def test_load_file(self):
        example = Example(
            iteration=0,
            dataset_idx=0,
            dataset_name="test",
            doc_ids=[0],
            text="test",
            token_ids=[0],
        )
        with tempfile.NamedTemporaryFile(mode="wb") as temp:
            with gzip.open(temp.name, "wt") as f:
                f.write(json.dumps(dataclasses.asdict(example)) + "\n")
                f.write(json.dumps(dataclasses.asdict(example)) + "\n")
                f.write(json.dumps(dataclasses.asdict(example)) + "\n")
            temp.seek(0)
            examples = load_file(temp.name)
            self.assertEqual(len(examples), 3)
            self.assertEqual(examples[0].iteration, 0)
            self.assertEqual(examples[0].dataset_idx, 0)
            self.assertEqual(examples[0].dataset_name, "test")
            self.assertEqual(examples[0].doc_ids, [0])
            self.assertEqual(examples[0].text, "test")
            self.assertEqual(examples[0].token_ids, [0])
