import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Union

# FOLDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "refined"]
FOLDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]
LOCAL_RANKS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

PREFIX_LENGTHS = [100, 200, 500, 1_000]
COMPLETION_LENGTH = 50


@dataclass
class Example:
    iteration: int
    dataset_idx: int
    dataset_name: str
    doc_ids: list[int]
    text: str
    token_ids: list[int]

    prefix_stats: dict[int, dict[str, int]] = field(default_factory=dict)
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


def save_examples(examples: Iterable[Example], path: Union[str, Path]) -> None:
    """Save examples in JSON format.

    Args:
        examples (Iterable[Example]): An iterable over the examples.
        path (Union[str, Path]): Path to the file.
    """
    with gzip.open(path, "wt") as f:
        for example in examples:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
