# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict as dataclass_asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from datasets import __version__ as datasets_version
from transformers import __version__ as transformers_version

from ..extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

RESUME_FORMAT_VERSION = 2
RESUME_SUFFIX = ".inprogress"
ASSEMBLY_SUFFIX = ".assembling"
SHARD_METADATA_FILE = "resume_shard.json"
MEDIA_COLUMNS = ("_images", "_videos", "_audios")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclass_asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return repr(value)


def fingerprint_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(_jsonable(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_implementation_digest(*objects: object) -> str:
    implementation_files = {os.path.abspath(__file__): {__name__}}
    for obj in objects:
        try:
            source_file = inspect.getsourcefile(type(obj))
        except (TypeError, OSError):
            source_file = None
        if source_file is not None:
            source_file = os.path.abspath(source_file)
            identifier = f"{type(obj).__module__}.{type(obj).__qualname__}"
            implementation_files.setdefault(source_file, set()).add(identifier)

    digest = hashlib.sha256()
    for source_file, identifiers in sorted(implementation_files.items(), key=lambda item: sorted(item[1])):
        digest.update("\n".join(sorted(identifiers)).encode("utf-8"))
        with open(source_file, "rb") as input_file:
            digest.update(input_file.read())
    return digest.hexdigest()


def _get_tokenizer_semantics_digest(tokenizer: "PreTrainedTokenizer") -> str:
    """Hash tokenizer semantics once per process, independently of asset paths."""
    cached = getattr(tokenizer, "_speechlmm_semantics_digest", None)
    if isinstance(cached, str):
        return cached

    digest = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        digest.update(backend.to_str().encode("utf-8"))
    else:
        digest.update(
            json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    digest.update(str(getattr(tokenizer, "chat_template", None)).encode("utf-8"))
    value = digest.hexdigest()
    setattr(tokenizer, "_speechlmm_semantics_digest", value)
    return value


def get_preprocessing_fingerprint(
    data_args: "DataArguments",
    stage: str,
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    dataset_processor: "DatasetProcessor",
) -> str:
    r"""Fingerprint every setting that can change tokenized output."""
    data_config = data_args.to_dict()
    for key in [
        "overwrite_cache",
        "preprocessing_batch_size",
        "preprocessing_num_workers",
        "preprocessing_resume",
        "preprocessing_shard_size",
        "tokenized_path",
        "sample_shape_index_path",
        "dynamic_batching",
        "dynamic_batching_memory_profile",
        "dynamic_batching_index",
        "dynamic_batching_target_memory_used",
        "dynamic_batching_candidate_fraction",
        "dynamic_batching_oversize_policy",
        "interleave_anchor_dataset",
    ]:
        data_config.pop(key, None)

    tokenizer_config = {
        "class": type(tokenizer).__qualname__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "added_vocab": tokenizer.get_added_vocab(),
        "semantics_digest": _get_tokenizer_semantics_digest(tokenizer),
    }
    processor_config = None
    if processor is not None:
        processor_config = {
            "class": type(processor).__qualname__,
            "config": processor.to_dict() if hasattr(processor, "to_dict") else repr(processor),
        }

    return fingerprint_payload(
        {
            "format_version": 1,
            "stage": stage,
            "data": data_config,
            "template": template,
            "tokenizer": tokenizer_config,
            "processor": processor_config,
            "datasets_version": datasets_version,
            "transformers_version": transformers_version,
            "implementation": _get_implementation_digest(dataset_processor, template, template.mm_plugin),
        }
    )


def get_resume_path(output_path: str | os.PathLike[str]) -> Path:
    return Path(f"{output_path}{RESUME_SUFFIX}")


def get_resume_lock_path(output_path: str | os.PathLike[str]) -> Path:
    return Path(f"{output_path}{RESUME_SUFFIX}.lock")


def _get_dataset_fingerprint(dataset: Dataset) -> str:
    r"""Keep the private datasets fingerprint dependency in one compatibility boundary."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str):
        raise RuntimeError("The installed datasets version does not expose a dataset fingerprint.")
    return fingerprint


def _iter_media_paths(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            yield path
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_media_paths(item)


def _get_media_fingerprint(
    dataset: Dataset,
    stat_cache: dict[str, tuple[int, int] | None] | None = None,
) -> str:
    r"""Fingerprint external media identities so changed files invalidate completed shards."""
    digest = hashlib.sha256()
    stat_cache = {} if stat_cache is None else stat_cache
    for column_name in MEDIA_COLUMNS:
        if column_name not in dataset.column_names:
            continue
        if dataset.data.column(column_name).null_count == len(dataset):
            continue

        digest.update(column_name.encode("utf-8"))
        for row_index, value in enumerate(dataset[column_name]):
            for media_path in _iter_media_paths(value):
                if media_path not in stat_cache:
                    try:
                        stat = os.stat(media_path)
                        stat_cache[media_path] = (stat.st_size, stat.st_mtime_ns)
                    except OSError:
                        stat_cache[media_path] = None

                digest.update(f"{row_index}:{media_path}:".encode())
                identity = stat_cache[media_path]
                digest.update(b"missing" if identity is None else f"{identity[0]}:{identity[1]}".encode("ascii"))

    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_completed_shard(shard_path: Path, expected_metadata: dict[str, Any]) -> Dataset | None:
    metadata_path = shard_path / SHARD_METADATA_FILE
    if not shard_path.exists():
        return None
    if not shard_path.is_dir() or not metadata_path.is_file():
        logger.warning_rank0(f"Discarding uncommitted preprocessing shard {shard_path}.")
        if shard_path.is_dir():
            shutil.rmtree(shard_path, ignore_errors=True)
        else:
            shard_path.unlink(missing_ok=True)
        return None

    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"Shard metadata mismatch for {key}.")

        dataset = load_from_disk(str(shard_path))
        if len(dataset) != metadata["output_rows"]:
            raise ValueError("Saved shard row count does not match its completion metadata.")
        return dataset
    except Exception as error:
        logger.warning_rank0(f"Discarding incomplete preprocessing shard {shard_path}: {error}")
        shutil.rmtree(shard_path, ignore_errors=True)
        return None


def process_dataset_in_resumable_shards(
    dataset: Dataset,
    process_shard: Callable[[Dataset], Dataset],
    output_path: str | os.PathLike[str],
    split_name: str,
    shard_size: int,
    preprocessing_fingerprint: str,
) -> Dataset:
    if shard_size <= 0:
        raise ValueError("preprocessing_shard_size must be greater than zero.")

    num_shards = max(1, (len(dataset) + shard_size - 1) // shard_size)
    split_path = get_resume_path(output_path) / split_name
    split_path.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _get_dataset_fingerprint(dataset)
    manifest = {
        "format_version": RESUME_FORMAT_VERSION,
        "split": split_name,
        "source_fingerprint": source_fingerprint,
        "source_rows": len(dataset),
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "shard_size": shard_size,
        "num_shards": num_shards,
    }
    manifest_path = split_path / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as manifest_file:
            saved_manifest = json.load(manifest_file)
        if saved_manifest != manifest:
            raise ValueError(
                f"Cannot resume tokenization from {split_path}: the source data or preprocessing configuration changed. "
                "Use a new tokenized_path or remove the stale in-progress directory."
            )
    else:
        _write_json_atomic(manifest_path, manifest)

    for temporary_path in split_path.glob(".shard-*.tmp-*"):
        shutil.rmtree(temporary_path, ignore_errors=True)

    processed_shards = []
    media_stat_cache: dict[str, tuple[int, int] | None] = {}
    for shard_index in range(num_shards):
        start = shard_index * shard_size
        stop = min(start + shard_size, len(dataset))
        shard_name = f"shard-{shard_index:05d}-of-{num_shards:05d}"
        shard_path = split_path / shard_name
        input_shard = dataset.select(range(start, stop))
        expected_metadata = {
            "format_version": RESUME_FORMAT_VERSION,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "source_fingerprint": source_fingerprint,
            "media_fingerprint": _get_media_fingerprint(input_shard, media_stat_cache),
            "shard_index": shard_index,
            "input_start": start,
            "input_stop": stop,
        }
        completed_shard = _load_completed_shard(shard_path, expected_metadata)
        if completed_shard is not None:
            logger.info_rank0(f"Reusing completed preprocessing shard {shard_index + 1}/{num_shards}.")
            processed_shards.append(completed_shard)
            continue

        logger.info_rank0(f"Processing preprocessing shard {shard_index + 1}/{num_shards} ({start}:{stop}).")
        temporary_path = Path(tempfile.mkdtemp(prefix=f".{shard_name}.tmp-", dir=split_path))
        try:
            processed_shard = process_shard(input_shard)
            processed_shard.save_to_disk(str(temporary_path), num_shards=1)
            shard_metadata = {**expected_metadata, "output_rows": len(processed_shard)}
            _write_json_atomic(temporary_path / SHARD_METADATA_FILE, shard_metadata)
            os.replace(temporary_path, shard_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        processed_shards.append(load_from_disk(str(shard_path)))

    if len(processed_shards) == 1:
        return processed_shards[0]
    return concatenate_datasets(processed_shards)


def _link_dataset_files(dataset: Dataset, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cache_files = [Path(cache_file["filename"]) for cache_file in dataset.cache_files]
    # Column removal can leave a logical Dataset view over an Arrow file with a
    # wider physical schema. Linking that file with the view's metadata creates
    # an unloadable artifact, so transformed views must be materialized.
    physical_features = Dataset.from_file(str(cache_files[0])).features if cache_files else None
    if not cache_files or physical_features != dataset.features:
        dataset.save_to_disk(str(destination))
        return

    dataset_format = dataset.format
    state = {
        "_fingerprint": _get_dataset_fingerprint(dataset),
        "_format_columns": dataset_format["columns"],
        "_format_kwargs": dataset_format["format_kwargs"],
        "_format_type": dataset_format["type"],
        "_output_all_columns": dataset_format["output_all_columns"],
    }
    state["_split"] = str(dataset.split) if dataset.split is not None else None
    state["_data_files"] = []
    num_files = len(cache_files)
    for file_index, source in enumerate(cache_files):
        filename = f"data-{file_index:05d}-of-{num_files:05d}.arrow"
        target = destination / filename
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        state["_data_files"].append({"filename": filename})

    with (destination / "state.json").open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)

    dataset.info.write_to_directory(str(destination), pretty_print=True)


def finalize_resumable_dataset_dict(
    dataset_dict: DatasetDict,
    output_path: str | os.PathLike[str],
    prepare_assembly: Callable[[Path], None] | None = None,
) -> DatasetDict:
    output_path = Path(output_path)
    assembly_path = Path(f"{output_path}{ASSEMBLY_SUFFIX}")
    if output_path.exists():
        raise FileExistsError(f"Cannot finalize tokenized data because {output_path} already exists.")

    if assembly_path.exists():
        shutil.rmtree(assembly_path)
    assembly_path.parent.mkdir(parents=True, exist_ok=True)
    assembly_path.mkdir()

    try:
        for split_name, dataset in dataset_dict.items():
            _link_dataset_files(dataset, assembly_path / split_name)
        with (assembly_path / "dataset_dict.json").open("w", encoding="utf-8") as dataset_dict_file:
            json.dump({"splits": list(dataset_dict)}, dataset_dict_file)

        if prepare_assembly is not None:
            prepare_assembly(assembly_path)

        validated_dataset = load_from_disk(str(assembly_path))
        if set(validated_dataset) != set(dataset_dict):
            raise RuntimeError("Finalized tokenized dataset has different splits.")
        for split_name, dataset in dataset_dict.items():
            if len(validated_dataset[split_name]) != len(dataset):
                raise RuntimeError(f"Finalized split {split_name} has a different row count.")
            if validated_dataset[split_name].features != dataset.features:
                raise RuntimeError(f"Finalized split {split_name} has different features.")

        os.replace(assembly_path, output_path)
        finalized_dataset = load_from_disk(str(output_path))
        shutil.rmtree(get_resume_path(output_path), ignore_errors=True)
        return finalized_dataset
    except Exception:
        shutil.rmtree(assembly_path, ignore_errors=True)
        raise
