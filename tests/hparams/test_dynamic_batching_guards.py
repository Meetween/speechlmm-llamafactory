# Copyright 2025 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

from pathlib import Path

import pytest

from llamafactory.hparams import parser as parser_module
from llamafactory.hparams.data_args import DataArguments


def test_build_sample_shape_index_requires_tokenized_path():
    with pytest.raises(ValueError, match="requires tokenized_path"):
        DataArguments(dataset="demo", build_sample_shape_index=True, tokenized_path=None)


def test_build_sample_shape_index_rejects_dynamic_batching_same_command():
    with pytest.raises(ValueError, match="before enabling dynamic_batching"):
        DataArguments(
            dataset="demo",
            tokenized_path="/tmp/prepared",
            build_sample_shape_index=True,
            dynamic_batching=True,
            dynamic_batching_memory_profile="/tmp/profile.json",
            dynamic_batching_index="/tmp/prepared/sample_shapes",
        )


def test_parser_source_rejects_eval_and_epochs_with_dynamic_batching():
    source = Path(parser_module.__file__).read_text(encoding="utf-8")
    assert "dynamic batching is step-based; remove num_train_epochs" in source
    assert "do_eval/predict_with_generate" in source
    assert 'finetuning_args.stage != "sft"' in source
