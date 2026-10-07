"""Per-turn record passed between the agent loop, the evaluator and the judge.

The fields this harness uses from slime's `Sample` (OpenWebRL 05f8ed4, slime/utils/types.py).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Sample:
    index: int | None = None
    prompt: str | list[dict[str, str]] = ""
    multimodal_inputs: dict[str, Any] | None = None  # images, e.g. the judge's screenshots
    response: str = ""
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
