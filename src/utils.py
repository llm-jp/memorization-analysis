import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

FOLDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "refined"]
LOCAL_RANKS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]


@dataclass
class Example:
    iteration: int
    dataset_idx: int
    dataset_name: str
    doc_ids: list[int]
    text: str
    token_ids: list[int]


def load_examples(path: Union[str, Path]) -> dict[int, list[Example]]:
    """Load a file containing examples in JSON format.

    Args:
        path (Union[str, Path]): Path to the file.

    Returns:
        dict[int, list[Example]: A mapping from iteration to a list of examples.
    """
    step_examples_map = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            example = Example(**json.loads(line))
            step_examples_map[example.iteration].append(example)
    return step_examples_map
