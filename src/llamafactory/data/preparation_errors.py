# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Row-local failures that may be quarantined during prepared-data creation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NoReturn

from datasets import Dataset


ALIGNMENT_ERROR_COLUMN = "_sample_shape_alignment_error"
PROCESSING_ERROR_COLUMN = "_sample_shape_processing_error"


@dataclass
class PreprocessedDatasetResult:
    dataset: Dataset
    rejected_samples: list[dict[str, Any]]


class RecoverablePreparationError(Exception):
    def __init__(self, message: str, *, processing_stage: str, exception_class: str | None = None) -> None:
        super().__init__(message)
        self.processing_stage = processing_stage
        self.exception_class = exception_class or type(self).__name__

    @classmethod
    def from_exception(cls, error: Exception, processing_stage: str) -> RecoverablePreparationError:
        return cls(
            str(error),
            processing_stage=processing_stage,
            exception_class=type(error).__name__,
        )


def is_resource_exhaustion_error(error: BaseException) -> bool:
    """Resource failures must abort; treating them as bad rows would hide capacity bugs."""
    message = str(error).lower()
    return (
        isinstance(error, MemoryError)
        or type(error).__name__ == "OutOfMemoryError"
        or "out of memory" in message
        or "cannot allocate memory" in message
        or "can't allocate memory" in message
    )


def raise_recoverable_preparation_error(error: Exception, processing_stage: str) -> NoReturn:
    if is_resource_exhaustion_error(error):
        raise error
    raise RecoverablePreparationError.from_exception(error, processing_stage) from error


def serialize_alignment_error(error: Exception) -> str:
    return json.dumps(
        {
            "exception_class": type(error).__name__,
            "exception_message": str(error),
            "processing_stage": "dataset_alignment",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def deserialize_error(value: str) -> dict[str, Any]:
    return json.loads(value)


def build_rejection_record(
    examples: dict[str, list[Any]],
    index: int,
    error: RecoverablePreparationError,
    *,
    source_column: str,
    sample_id_column: str,
) -> dict[str, Any]:
    return {
        "source": str(examples.get(source_column, ["unknown"])[index]),
        "sample_id": str(examples.get(sample_id_column, [index])[index]),
        "reason": "preprocessing_error",
        "processing_stage": error.processing_stage,
        "exception_class": error.exception_class,
        "exception_message": str(error),
    }


def serialize_rejection(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
