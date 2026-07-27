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


def test_dynamic_evaluation_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_eval_samples_per_dataset"):
        DataArguments(
            dataset="demo",
            tokenized_path="/tmp/prepared",
            dynamic_batching=True,
            dynamic_batching_memory_profile="/tmp/profile.json",
            dynamic_batching_index="/tmp/prepared/sample_shapes",
            dynamic_batching_max_eval_samples_per_dataset=0,
        )


def test_parser_source_accepts_integer_epochs_and_rejects_non_loss_eval():
    source = Path(parser_module.__file__).read_text(encoding="utf-8")
    assert "requires a positive integer num_train_epochs" in source
    assert "accepts either max_steps or num_train_epochs, not both" in source
    assert "requires prediction_loss_only=true" in source
    assert "per_device_eval_batch_size=1" not in source
    assert 'finetuning_args.stage != "sft"' in source
