import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union

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

    metrics: dict[str, float] = field(default_factory=dict)


def load_examples(path: Union[str, Path]) -> Iterator[Example]:
    """Load a file containing examples in JSON format.

    Args:
        path (Union[str, Path]): Path to the file.

    Yields:
        Iterator[Example]: An iterator over the examples.
    """
    with gzip.open(path, "rt") as f:
        for line in f:
            yield Example(**json.loads(line))
