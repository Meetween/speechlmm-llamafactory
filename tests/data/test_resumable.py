import json
import multiprocessing
import os

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from llamafactory.data.resumable import (
    finalize_resumable_dataset_dict,
    get_resume_path,
    process_dataset_in_resumable_shards,
)
from llamafactory.extras.misc import has_tokenized_data


def _add_doubled_column(dataset: Dataset) -> Dataset:
    return dataset.map(lambda batch: {"doubled": [value * 2 for value in batch["value"]]}, batched=True)


def _terminate_during_second_shard(output_path: str) -> None:
    source = Dataset.from_dict({"value": list(range(6))})

    def terminate(shard: Dataset) -> Dataset:
        if shard[0]["value"] == 2:
            os._exit(17)
        return _add_doubled_column(shard)

    process_dataset_in_resumable_shards(
        source,
        terminate,
        output_path,
        "train",
        shard_size=2,
        preprocessing_fingerprint="process-death-test",
    )


def test_resumable_preprocessing_skips_completed_shards(tmp_path):
    source = Dataset.from_dict({"value": list(range(10))})
    output_path = tmp_path / "tokenized"
    attempted_starts = []

    def interrupt_second_shard(shard: Dataset) -> Dataset:
        attempted_starts.append(shard[0]["value"])
        if len(attempted_starts) == 2:
            raise RuntimeError("simulated interruption")
        return _add_doubled_column(shard)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        process_dataset_in_resumable_shards(
            source,
            interrupt_second_shard,
            output_path,
            "train",
            shard_size=4,
            preprocessing_fingerprint="same-configuration",
        )

    resume_path = get_resume_path(output_path)
    assert (resume_path / "train" / "shard-00000-of-00003" / "resume_shard.json").is_file()
    assert not has_tokenized_data(output_path)

    resumed_starts = []

    def record_resumed_shards(shard: Dataset) -> Dataset:
        resumed_starts.append(shard[0]["value"])
        return _add_doubled_column(shard)

    processed = process_dataset_in_resumable_shards(
        source,
        record_resumed_shards,
        output_path,
        "train",
        shard_size=4,
        preprocessing_fingerprint="same-configuration",
    )

    assert resumed_starts == [4, 8]
    assert processed["value"] == list(range(10))
    assert processed["doubled"] == [value * 2 for value in range(10)]


def test_resumable_preprocessing_survives_process_death(tmp_path):
    output_path = tmp_path / "tokenized"
    process = multiprocessing.get_context("spawn").Process(
        target=_terminate_during_second_shard,
        args=(str(output_path),),
    )
    process.start()
    # Spawn + datasets I/O can exceed 30s under a loaded suite on shared FS.
    process.join(timeout=180)
    if process.is_alive():
        process.kill()
        process.join(timeout=30)
        pytest.fail("child process did not exit within 180s")

    assert process.exitcode == 17
    resume_path = get_resume_path(output_path)
    assert (resume_path / "train" / "shard-00000-of-00003" / "resume_shard.json").is_file()
    resumed_starts = []

    def record(shard):
        resumed_starts.append(shard[0]["value"])
        return _add_doubled_column(shard)

    resumed = process_dataset_in_resumable_shards(
        Dataset.from_dict({"value": list(range(6))}),
        record,
        output_path,
        "train",
        shard_size=2,
        preprocessing_fingerprint="process-death-test",
    )

    assert resumed_starts == [2, 4]
    assert resumed["doubled"] == [0, 2, 4, 6, 8, 10]


def test_resumable_preprocessing_rejects_changed_configuration(tmp_path):
    source = Dataset.from_dict({"value": list(range(5))})
    output_path = tmp_path / "tokenized"
    process_dataset_in_resumable_shards(
        source,
        _add_doubled_column,
        output_path,
        "train",
        shard_size=2,
        preprocessing_fingerprint="first-configuration",
    )

    with pytest.raises(ValueError, match="source data or preprocessing configuration changed"):
        process_dataset_in_resumable_shards(
            source,
            _add_doubled_column,
            output_path,
            "train",
            shard_size=2,
            preprocessing_fingerprint="changed-configuration",
        )


def test_resumable_preprocessing_repairs_uncommitted_shard(tmp_path):
    source = Dataset.from_dict({"value": list(range(3))})
    output_path = tmp_path / "tokenized"
    incomplete_shard = get_resume_path(output_path) / "train" / "shard-00000-of-00001"
    incomplete_shard.mkdir(parents=True)
    (incomplete_shard / "partial.arrow").touch()

    processed = process_dataset_in_resumable_shards(
        source,
        _add_doubled_column,
        output_path,
        "train",
        shard_size=3,
        preprocessing_fingerprint="repair-test",
    )

    assert processed["doubled"] == [0, 2, 4]
    assert (incomplete_shard / "resume_shard.json").is_file()


def test_resumable_preprocessing_reprocesses_changed_media(tmp_path):
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"first")
    source = Dataset.from_dict({"value": [1], "_audios": [[str(media_path)]]})
    output_path = tmp_path / "tokenized"
    processed_shards = []

    def record_processed_shard(shard: Dataset) -> Dataset:
        processed_shards.append(shard[0]["value"])
        return _add_doubled_column(shard)

    process_dataset_in_resumable_shards(
        source,
        record_processed_shard,
        output_path,
        "train",
        shard_size=1,
        preprocessing_fingerprint="media-test",
    )
    media_path.write_bytes(b"changed-media")
    process_dataset_in_resumable_shards(
        source,
        record_processed_shard,
        output_path,
        "train",
        shard_size=1,
        preprocessing_fingerprint="media-test",
    )

    assert processed_shards == [1, 1]


def test_resumable_finalization_produces_standard_dataset_dict(tmp_path):
    source = Dataset.from_dict({"value": list(range(7))})
    output_path = tmp_path / "tokenized"
    processed = process_dataset_in_resumable_shards(
        source,
        _add_doubled_column,
        output_path,
        "train",
        shard_size=3,
        preprocessing_fingerprint="finalization-test",
    )
    finalized = finalize_resumable_dataset_dict(DatasetDict({"train": processed}), output_path)
    loaded = load_from_disk(str(output_path))

    assert has_tokenized_data(output_path)
    assert not get_resume_path(output_path).exists()
    assert finalized["train"].data.table.equals(processed.data.table)
    assert loaded["train"].data.table.equals(processed.data.table)
    assert len(list((output_path / "train").glob("*.arrow"))) == 3


def test_resumable_finalization_materializes_transformed_view_and_callback(tmp_path):
    source = Dataset.from_dict({"value": list(range(5))})
    output_path = tmp_path / "tokenized"
    processed = process_dataset_in_resumable_shards(
        source,
        _add_doubled_column,
        output_path,
        "train",
        shard_size=2,
        preprocessing_fingerprint="transformed-view-test",
    )
    clean_view = processed.remove_columns("doubled")

    def add_sidecar(assembly_path):
        sidecar = assembly_path / "sample_shapes"
        sidecar.mkdir()
        (sidecar / "manifest.json").write_text("{}", encoding="utf-8")

    finalize_resumable_dataset_dict(
        DatasetDict({"train": clean_view}),
        output_path,
        prepare_assembly=add_sidecar,
    )
    loaded = load_from_disk(str(output_path))

    assert loaded["train"].column_names == ["value"]
    assert loaded["train"]["value"] == list(range(5))
    assert (output_path / "sample_shapes" / "manifest.json").is_file()


def test_resumable_finalization_recovers_after_sidecar_failure(tmp_path):
    source = Dataset.from_dict({"value": list(range(4))})
    output_path = tmp_path / "tokenized"
    processed = process_dataset_in_resumable_shards(
        source,
        _add_doubled_column,
        output_path,
        "train",
        shard_size=2,
        preprocessing_fingerprint="publication-failure-test",
    )

    def fail_sidecar(_assembly_path):
        raise RuntimeError("simulated sidecar failure")

    with pytest.raises(RuntimeError, match="simulated sidecar failure"):
        finalize_resumable_dataset_dict(
            DatasetDict({"train": processed}),
            output_path,
            prepare_assembly=fail_sidecar,
        )

    assert not output_path.exists()
    assert not (tmp_path / "tokenized.assembling").exists()
    assert get_resume_path(output_path).exists()
    finalized = finalize_resumable_dataset_dict(DatasetDict({"train": processed}), output_path)
    assert finalized["train"]["value"] == list(range(4))


def test_has_tokenized_data_rejects_incomplete_dataset_dict(tmp_path):
    output_path = tmp_path / "partial"
    output_path.mkdir()
    (output_path / "dataset_dict.json").write_text(json.dumps({"splits": ["train"]}), encoding="utf-8")

    assert not has_tokenized_data(output_path)
