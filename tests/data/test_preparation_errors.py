from types import SimpleNamespace

import pytest
from datasets import Dataset
from speechlmm.data_loading_optimization.sample_shapes import SAMPLE_ID_COLUMN, SOURCE_ID_COLUMN

from llamafactory.data.loader import _get_preprocessed_dataset
from llamafactory.data.preparation_errors import (
    PROCESSING_ERROR_COLUMN,
    PreprocessedDatasetResult,
    RecoverablePreparationError,
    is_resource_exhaustion_error,
)
from llamafactory.data.resumable import ResumableDatasetResult, process_dataset_in_resumable_shards
from llamafactory.hparams import DataArguments


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens=False):
        return str(token_ids)


class FakeMMPlugin:
    def process_messages_with_layout(self, messages, images, videos, audios, processor):
        content = messages[0]["content"]
        if content == "recoverable":
            raise RecoverablePreparationError(
                "broken row",
                processing_stage="audio_placeholder_expansion",
            )
        if content == "systemic":
            raise RuntimeError("simulated repository bug")
        return messages, SimpleNamespace(audios=(), images=(), videos=())

    def process_messages(self, messages, images, videos, audios, processor):
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        return input_ids, labels


class FakeTemplate:
    efficient_eos = False
    mm_plugin = FakeMMPlugin()

    def encode_multiturn_batch(self, tokenizer, batch_messages, systems, tools):
        return [[([11], [12])] for _ in batch_messages]

    def encode_multiturn(self, tokenizer, messages, system, tools):
        return [([11], [12])]


def _source(contents):
    size = len(contents)
    return Dataset.from_dict(
        {
            "_prompt": [[{"role": "user", "content": content}] for content in contents],
            "_response": [[{"role": "assistant", "content": "answer"}] for _ in contents],
            "_system": [""] * size,
            "_tools": [""] * size,
            "_images": [None] * size,
            "_videos": [None] * size,
            "_audios": [None] * size,
            "_codec_tokens": [None] * size,
            SOURCE_ID_COLUMN: ["source"] * size,
            SAMPLE_ID_COLUMN: [f"source:{index}" for index in range(size)],
            "_sample_shape_alignment_error": [None] * size,
        }
    )


def _data_args(tmp_path):
    return DataArguments(
        build_sample_shape_index=True,
        tokenized_path=str(tmp_path / "tokenized"),
        preprocessing_num_workers=None,
        preprocessing_batch_size=8,
    )


def _training_args():
    return SimpleNamespace(predict_with_generate=False, local_process_index=0, should_log=False)


def test_prepared_tokenization_quarantines_only_recoverable_rows(tmp_path):
    result = _get_preprocessed_dataset(
        _source(["valid", "recoverable", "valid"]),
        _data_args(tmp_path),
        _training_args(),
        "sft",
        FakeTemplate(),
        FakeTokenizer(),
        return_result=True,
    )

    assert isinstance(result, PreprocessedDatasetResult)
    assert result.dataset[SAMPLE_ID_COLUMN] == ["source:0", "source:2"]
    assert PROCESSING_ERROR_COLUMN not in result.dataset.column_names
    assert [record["sample_id"] for record in result.rejected_samples] == ["source:1"]


def test_prepared_tokenization_does_not_hide_unexpected_errors(tmp_path):
    with pytest.raises(RuntimeError, match="simulated repository bug"):
        _get_preprocessed_dataset(
            _source(["systemic"]),
            _data_args(tmp_path),
            _training_args(),
            "sft",
            FakeTemplate(),
            FakeTokenizer(),
            return_result=True,
        )


def test_existing_row_rejection_does_not_hide_batch_tokenizer_failure(tmp_path):
    class BrokenBatchTemplate(FakeTemplate):
        def encode_multiturn_batch(self, tokenizer, batch_messages, systems, tools):
            raise ValueError("simulated batch tokenizer bug")

    with pytest.raises(ValueError, match="simulated batch tokenizer bug"):
        _get_preprocessed_dataset(
            _source(["recoverable", "valid"]),
            _data_args(tmp_path),
            _training_args(),
            "sft",
            BrokenBatchTemplate(),
            FakeTokenizer(),
            return_result=True,
        )


def test_resource_exhaustion_is_never_a_row_rejection():
    assert is_resource_exhaustion_error(MemoryError("host allocation failed"))
    assert is_resource_exhaustion_error(RuntimeError("cannot allocate memory"))
    assert not is_resource_exhaustion_error(RuntimeError("decoder failed"))


def test_resumable_rejections_are_reused_without_duplication(tmp_path):
    source = Dataset.from_dict({"value": list(range(5))})
    calls = []

    def process(shard):
        calls.append(shard[0]["value"])
        rejected = [
            {
                "source": "source",
                "sample_id": f"source:{value}",
                "reason": "preprocessing_error",
            }
            for value in shard["value"]
            if value == 1
        ]
        valid = shard.filter(lambda value: value != 1, input_columns=["value"])
        return PreprocessedDatasetResult(valid, rejected)

    first = process_dataset_in_resumable_shards(
        source,
        process,
        tmp_path / "tokenized",
        "train",
        shard_size=2,
        preprocessing_fingerprint="same",
        return_result=True,
        require_row_accounting=True,
    )
    calls.clear()
    resumed = process_dataset_in_resumable_shards(
        source,
        process,
        tmp_path / "tokenized",
        "train",
        shard_size=2,
        preprocessing_fingerprint="same",
        return_result=True,
        require_row_accounting=True,
    )

    assert isinstance(first, ResumableDatasetResult)
    assert isinstance(resumed, ResumableDatasetResult)
    assert calls == []
    assert resumed.dataset["value"] == [0, 2, 3, 4]
    assert [record["sample_id"] for record in resumed.rejected_samples] == ["source:1"]
