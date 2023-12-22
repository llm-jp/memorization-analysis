import dataclasses
import gzip
import json
import tempfile
import unittest
from collections import defaultdict

from src.utils import Example, load_examples, save_examples


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

    def test_example_with_metrics(self):
        example = Example(
            iteration=0,
            dataset_idx=0,
            dataset_name="test",
            doc_ids=[0],
            text="test",
            token_ids=[0],
            metrics={"perplexity": 1.0},
        )
        self.assertEqual(example.iteration, 0)
        self.assertEqual(example.dataset_idx, 0)
        self.assertEqual(example.dataset_name, "test")
        self.assertEqual(example.doc_ids, [0])
        self.assertEqual(example.text, "test")
        self.assertEqual(example.token_ids, [0])
        self.assertEqual(example.metrics, {"perplexity": 1.0})

    def test_set_metrics(self):
        example = Example(
            iteration=0,
            dataset_idx=0,
            dataset_name="test",
            doc_ids=[0],
            text="test",
            token_ids=[0],
        )
        example.metrics["perplexity"] = 1.0
        self.assertEqual(example.metrics, {"perplexity": 1.0})


class TestLoadExamples(unittest.TestCase):
    def test_load_examples(self):
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

            step_examples_map = defaultdict(list)
            for example in load_examples(temp.name):
                step_examples_map[example.iteration].append(example)
            self.assertEqual(len(step_examples_map), 1)
            self.assertEqual(len(step_examples_map[0]), 3)
            self.assertEqual(step_examples_map[0][0].iteration, 0)
            self.assertEqual(step_examples_map[0][0].dataset_idx, 0)
            self.assertEqual(step_examples_map[0][0].dataset_name, "test")
            self.assertEqual(step_examples_map[0][0].doc_ids, [0])
            self.assertEqual(step_examples_map[0][0].text, "test")
            self.assertEqual(step_examples_map[0][0].token_ids, [0])


class TestSaveExamples(unittest.TestCase):
    def test_save_examples(self):
        example = Example(
            iteration=0,
            dataset_idx=0,
            dataset_name="test",
            doc_ids=[0],
            text="test",
            token_ids=[0],
        )
        with tempfile.NamedTemporaryFile(mode="w") as temp:
            save_examples([example, example, example], temp.name)
            temp.seek(0)

            step_examples_map = defaultdict(list)
            for example in load_examples(temp.name):
                step_examples_map[example.iteration].append(example)
            self.assertEqual(len(step_examples_map), 1)
            self.assertEqual(len(step_examples_map[0]), 3)
            self.assertEqual(step_examples_map[0][0].iteration, 0)
            self.assertEqual(step_examples_map[0][0].dataset_idx, 0)
            self.assertEqual(step_examples_map[0][0].dataset_name, "test")
            self.assertEqual(step_examples_map[0][0].doc_ids, [0])
            self.assertEqual(step_examples_map[0][0].text, "test")
            self.assertEqual(step_examples_map[0][0].token_ids, [0])
