from dataclasses import dataclass
from typing import Any


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
