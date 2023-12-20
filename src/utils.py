import json
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass
class Example:
    iteration: int
    dataset_idx: int
    dataset_name: str
    doc_ids: list[int]
    text: str
    token_ids: list[int]

    @classmethod
    def from_json(cls, json_obj: dict[str, Any]):
        return cls(**json_obj)


def load_file(path: Union[str, Path]) -> list[Example]:
    """Load a file containing examples in JSON format.

    Args:
        path (Union[str, Path]): Path to the file.

    Returns:
        list[Example]: List of examples.
    """
    with gzip.open(path, "rt") as f:
        return [Example.from_json(json.loads(line)) for line in f]
