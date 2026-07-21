# Copyright 2025 the LlamaFactory team.
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
import os
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub.utils import WeakFileLock
from speechlmm.data_loading_optimization.sample_shapes import (
    SAMPLE_ID_COLUMN,
    SAMPLE_SHAPE_FEATURES,
    SOURCE_ID_COLUMN,
)
from speechlmm.data_loading_optimization.validation import (
    REJECTED_SAMPLES_FILE,
    quarantine_context_overflow,
    rejected_samples_summary,
    validate_dataset_media,
    write_rejected_samples,
)

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .preparation_errors import (
    PROCESSING_ERROR_COLUMN,
    PreprocessedDatasetResult,
    deserialize_error,
)
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)
from .resumable import (
    ResumableDatasetResult,
    finalize_resumable_dataset_dict,
    get_preprocessing_fingerprint,
    get_resume_lock_path,
    process_dataset_in_resumable_shards,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

PREPARED_MEDIA_BATCH_SIZE = 32
PREPARED_MEDIA_MAX_WORKERS = 4
PREPARED_AUDIO_SHARD_SIZE = 1000


def _contains_media(dataset: "Dataset", column_name: str) -> bool:
    return column_name in dataset.column_names and dataset.data.column(column_name).null_count < len(dataset)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    source_name: str,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if data_args.build_sample_shape_index:
        reserved = {SOURCE_ID_COLUMN, SAMPLE_ID_COLUMN} & set(dataset.column_names)
        if reserved:
            raise ValueError(f"dataset {source_name!r} uses reserved sample-shape columns: {reserved}")
        dataset = dataset.add_column(SOURCE_ID_COLUMN, [source_name] * len(dataset))
        dataset = dataset.add_column(
            SAMPLE_ID_COLUMN,
            [f"{source_name}:{index}" for index in range(len(dataset))],
        )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        sampling_seed = int.from_bytes(
            hashlib.sha256(f"{training_args.seed}:{source_name}".encode()).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(sampling_seed)
        target_num = dataset_attr.num_samples
        indexes = rng.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = rng.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
    interleave_probs: list[float] | None = None,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        dataset = _load_single_dataset(dataset_attr, dataset_name, model_args, data_args, training_args)
        if data_args.build_sample_shape_index or data_args.dynamic_batching:
            identity_columns = {SOURCE_ID_COLUMN, SAMPLE_ID_COLUMN} & set(dataset.column_names)
            if not identity_columns:
                dataset = dataset.add_column(SOURCE_ID_COLUMN, [dataset_name] * len(dataset))
                dataset = dataset.add_column(
                    SAMPLE_ID_COLUMN,
                    [f"{dataset_name}:{index}" for index in range(len(dataset))],
                )
            elif identity_columns != {SOURCE_ID_COLUMN, SAMPLE_ID_COLUMN}:
                raise ValueError(
                    f"dataset {dataset_name!r} has incomplete sample identity columns: {identity_columns}"
                )

        datasets[dataset_name] = dataset

    if return_dict:
        return datasets
    elif data_args.build_sample_shape_index or data_args.dynamic_batching:
        # Preserve every source row exactly once. The CPU planner owns all
        # probability, exhaustion, and recycling semantics in dynamic mode.
        return concatenate_datasets(list(datasets.values()))
    else:
        return merge_dataset(
            list(datasets.values()), data_args, seed=training_args.seed, interleave_probs=interleave_probs
        )


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
    print_example: bool = True,
    keep_in_memory: bool = False,
    return_result: bool = False,
) -> Union["Dataset", "IterableDataset", "PreprocessedDatasetResult"] | None:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        batch_size = data_args.preprocessing_batch_size
        num_proc = data_args.preprocessing_num_workers
        if data_args.build_sample_shape_index and any(
            _contains_media(dataset, column) for column in ("_images", "_videos", "_audios")
        ):
            batch_size = min(batch_size, PREPARED_MEDIA_BATCH_SIZE)
            if num_proc is not None:
                num_proc = min(num_proc, PREPARED_MEDIA_MAX_WORKERS)
            if batch_size != data_args.preprocessing_batch_size or num_proc != data_args.preprocessing_num_workers:
                logger.warning_rank0_once(
                    "Prepared multimodal tokenization bounds each map batch to "
                    f"{batch_size} rows and preprocessing workers to {num_proc} to control host memory."
                )
        kwargs = dict(
            num_proc=num_proc,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            keep_in_memory=keep_in_memory,
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=(batch_size if not data_args.streaming else data_args.preprocessing_batch_size),
        remove_columns=column_names,
        **kwargs,
    )

    rejected_samples = []
    if data_args.build_sample_shape_index:
        errors = dataset[PROCESSING_ERROR_COLUMN]
        rejected_samples = [deserialize_error(error) for error in errors if error is not None]
        valid_indices = [index for index, error in enumerate(errors) if error is None]
        dataset = dataset.select(valid_indices).remove_columns(PROCESSING_ERROR_COLUMN)
        for column_name, feature in SAMPLE_SHAPE_FEATURES.items():
            dataset = dataset.cast_column(column_name, feature)

    if training_args.should_log and print_example:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    if return_result:
        return PreprocessedDatasetResult(dataset=dataset, rejected_samples=rejected_samples)
    return dataset


def _get_resumable_preprocessed_dataset(
    dataset: "Dataset",
    split_name: str,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> "ResumableDatasetResult":
    if data_args.tokenized_path is None:
        raise ValueError("tokenized_path is required for resumable preprocessing")

    dataset_processor = _get_dataset_processor(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=(training_args.predict_with_generate and is_eval),
    )
    preprocessing_fingerprint = get_preprocessing_fingerprint(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        dataset_processor,
    )
    shard_size = data_args.preprocessing_shard_size
    if data_args.build_sample_shape_index and _contains_media(dataset, "_audios"):
        shard_size = min(shard_size, PREPARED_AUDIO_SHARD_SIZE)
        if shard_size != data_args.preprocessing_shard_size:
            logger.warning_rank0_once(
                f"Prepared audio tokenization uses {shard_size}-row resume shards to limit replay and host memory."
            )

    result = process_dataset_in_resumable_shards(
        dataset=dataset,
        process_shard=lambda shard: _get_preprocessed_dataset(
            shard,
            data_args,
            training_args,
            stage,
            template,
            tokenizer,
            processor,
            is_eval,
            print_example=False,
            keep_in_memory=False,
            return_result=data_args.build_sample_shape_index,
        ),
        output_path=data_args.tokenized_path,
        split_name=split_name,
        shard_size=shard_size,
        preprocessing_fingerprint=preprocessing_fingerprint,
        return_result=True,
        require_row_accounting=data_args.build_sample_shape_index,
    )
    if not isinstance(result, ResumableDatasetResult):
        raise TypeError("resumable prepared-data processing did not return its summary")

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(result.dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")
    return result


def _validate_completed_prepared_data(
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
) -> None:
    if not data_args.build_sample_shape_index:
        return
    if not Path(data_args.sample_shape_index_path).is_dir():
        raise ValueError(
            "tokenized data already exists but its requested sample-shape index is missing; "
            "use a new tokenized_path or restore the complete prepared artifact"
        )

    from speechlmm.data_loading_optimization import load_sample_shape_index

    dataset_processor = _get_dataset_processor(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=training_args.predict_with_generate,
    )
    current_fingerprint = get_preprocessing_fingerprint(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        dataset_processor,
    )
    completed_index = load_sample_shape_index(data_args.sample_shape_index_path)
    if completed_index.preprocessing_fingerprint != current_fingerprint:
        raise ValueError(
            "completed prepared data was built with different tokenizer, processor, template, "
            "or preprocessing semantics; use a new tokenized_path"
        )


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    if data_args.build_sample_shape_index:
        resolved_limit = getattr(processor, "speechlmm_context_limit", None)
        if resolved_limit is None or int(resolved_limit) != int(data_args.cutoff_len):
            raise ValueError(
                "prepared-data tokenization requires the model-derived context limit; "
                "launch it through speechlmm.data.preprocess_dataset"
            )
    if processor is not None and data_args.max_input_audio_seconds is not None:
        setattr(processor, "max_input_audio_seconds", data_args.max_input_audio_seconds)

    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            _validate_completed_prepared_data(data_args, training_args, stage, template, tokenizer, processor)
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    if data_args.dynamic_batching:
        raise ValueError(
            "dynamic batching requires a previously tokenized dataset and sample-shape index; "
            "run speechlmm.data.preprocess_dataset first"
        )
    if data_args.build_sample_shape_index and stage != "sft":
        raise ValueError("sample-shape index building currently supports supervised fine-tuning data only")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(
            data_args.dataset,
            model_args,
            data_args,
            training_args,
            stage,
            interleave_probs=data_args.interleave_probs,
        )
        eval_interleave_probs = data_args.eval_interleave_probs
        if (
            eval_interleave_probs is None
            and data_args.interleave_probs is not None
            and data_args.eval_dataset is not None
            and len(data_args.eval_dataset) == len(data_args.interleave_probs)
        ):
            eval_interleave_probs = data_args.interleave_probs

        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
            interleave_probs=eval_interleave_probs,
        )

    output_lock = (
        WeakFileLock(get_resume_lock_path(data_args.tokenized_path))
        if data_args.preprocessing_resume and data_args.tokenized_path is not None
        else nullcontext()
    )
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        with output_lock:
            if data_args.tokenized_path is not None and has_tokenized_data(data_args.tokenized_path):
                _validate_completed_prepared_data(data_args, training_args, stage, template, tokenizer, processor)
                tokenized_data = load_from_disk(data_args.tokenized_path)
                logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
                return get_dataset_module(tokenized_data)

            train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
            preparation_rejections = []
            preparation_source_rows = sum(len(split) for split in train_dict.values()) + sum(
                len(split) for split in eval_dict.values()
            )
            if data_args.build_sample_shape_index:
                for key, split in list(train_dict.items()):
                    train_dict[key], rejected = validate_dataset_media(split, split=key)
                    preparation_rejections.extend(rejected)
                for key, split in list(eval_dict.items()):
                    eval_dict[key], rejected = validate_dataset_media(split, split=key)
                    preparation_rejections.extend(rejected)
                if "train" in train_dict and len(train_dict["train"]) == 0:
                    raise ValueError(
                        "all training samples failed media validation: "
                        f"{rejected_samples_summary(preparation_rejections)}"
                    )

            resumable = (
                data_args.preprocessing_resume and data_args.tokenized_path is not None and not data_args.packing
            )
            if data_args.preprocessing_resume and data_args.tokenized_path is not None and data_args.packing:
                logger.warning_rank0("Disabling resumable preprocessing because packing crosses shard boundaries.")

            if "train" in train_dict:
                if resumable:
                    result = _get_resumable_preprocessed_dataset(
                        train_dict["train"],
                        "train",
                        data_args,
                        training_args,
                        stage,
                        template,
                        tokenizer,
                        processor,
                        is_eval=False,
                    )
                    train_dict["train"] = result.dataset
                    preparation_rejections.extend(result.rejected_samples)
                else:
                    result = _get_preprocessed_dataset(
                        train_dict["train"],
                        data_args,
                        training_args,
                        stage,
                        template,
                        tokenizer,
                        processor,
                        is_eval=False,
                        return_result=data_args.build_sample_shape_index,
                    )
                    if isinstance(result, PreprocessedDatasetResult):
                        train_dict["train"] = result.dataset
                        preparation_rejections.extend(
                            {**record, "split": "train"} for record in result.rejected_samples
                        )
                    else:
                        train_dict["train"] = result

            for key in eval_dict:
                if resumable:
                    result = _get_resumable_preprocessed_dataset(
                        eval_dict[key],
                        key,
                        data_args,
                        training_args,
                        stage,
                        template,
                        tokenizer,
                        processor,
                        is_eval=True,
                    )
                    eval_dict[key] = result.dataset
                    preparation_rejections.extend(result.rejected_samples)
                else:
                    result = _get_preprocessed_dataset(
                        eval_dict[key],
                        data_args,
                        training_args,
                        stage,
                        template,
                        tokenizer,
                        processor,
                        is_eval=True,
                        return_result=data_args.build_sample_shape_index,
                    )
                    if isinstance(result, PreprocessedDatasetResult):
                        eval_dict[key] = result.dataset
                        preparation_rejections.extend({**record, "split": key} for record in result.rejected_samples)
                    else:
                        eval_dict[key] = result

            if data_args.build_sample_shape_index:
                for key, split in list(train_dict.items()):
                    train_dict[key], rejected = quarantine_context_overflow(split, split=key)
                    preparation_rejections.extend(rejected)
                for key, split in list(eval_dict.items()):
                    eval_dict[key], rejected = quarantine_context_overflow(split, split=key)
                    preparation_rejections.extend(rejected)
                if "train" in train_dict and len(train_dict["train"]) == 0:
                    raise ValueError(
                        "all training samples exceed the model context: "
                        f"{rejected_samples_summary(preparation_rejections)}"
                    )

                output_rows = sum(len(split) for split in train_dict.values()) + sum(
                    len(split) for split in eval_dict.values()
                )
                if preparation_source_rows != output_rows + len(preparation_rejections):
                    raise RuntimeError(
                        "prepared-data row accounting failed: "
                        f"source={preparation_source_rows}, output={output_rows}, "
                        f"rejected={len(preparation_rejections)}"
                    )
                if (
                    data_args.preprocessing_max_failed_samples is not None
                    and len(preparation_rejections) > data_args.preprocessing_max_failed_samples
                ):
                    raise RuntimeError(
                        f"preparation rejected {len(preparation_rejections)} rows, exceeding "
                        f"preprocessing_max_failed_samples={data_args.preprocessing_max_failed_samples}"
                    )
                if (
                    data_args.preprocessing_max_failed_fraction is not None
                    and len(preparation_rejections)
                    > data_args.preprocessing_max_failed_fraction * preparation_source_rows
                ):
                    raise RuntimeError(
                        f"preparation rejected {len(preparation_rejections)}/{preparation_source_rows} rows, "
                        "exceeding preprocessing_max_failed_fraction="
                        f"{data_args.preprocessing_max_failed_fraction}"
                    )

            shape_index = None
            shape_summary = None
            if data_args.build_sample_shape_index:
                from speechlmm.data_loading_optimization import (
                    TEMPORARY_SHAPE_COLUMNS,
                    build_sample_shape_index_from_dataset,
                    write_sample_shape_index,
                )

                tagged_train = train_dict.get("train")
                if tagged_train is None:
                    raise ValueError("sample-shape index building requires a training split")
                missing = set(TEMPORARY_SHAPE_COLUMNS) - set(tagged_train.column_names)
                if missing:
                    raise ValueError(f"sample geometry was lost during tokenization: {sorted(missing)}")
                dataset_processor = _get_dataset_processor(
                    data_args, stage, template, tokenizer, processor, do_generate=False
                )
                preprocessing_fingerprint = get_preprocessing_fingerprint(
                    data_args, stage, template, tokenizer, processor, dataset_processor
                )
                preparation_summary = rejected_samples_summary(preparation_rejections)
                shape_index, shape_summary = build_sample_shape_index_from_dataset(
                    dataset=tagged_train,
                    processor=processor,
                    preprocessing_fingerprint=preprocessing_fingerprint,
                    dataset_path=data_args.tokenized_path,
                    preparation_summary=preparation_summary,
                )
                train_dict["train"] = tagged_train.remove_columns(list(TEMPORARY_SHAPE_COLUMNS))
                for key, eval_split in eval_dict.items():
                    removable = [name for name in TEMPORARY_SHAPE_COLUMNS if name in eval_split.column_names]
                    if removable:
                        eval_dict[key] = eval_split.remove_columns(removable)

            dataset_dict = DatasetDict({**train_dict, **eval_dict})
            if data_args.tokenized_path is not None and training_args.should_save:
                if shape_index is not None:
                    output_root = Path(data_args.tokenized_path).resolve()
                    index_path = Path(data_args.sample_shape_index_path).resolve()
                    try:
                        relative_index_path = index_path.relative_to(output_root)
                    except ValueError as error:
                        raise ValueError(
                            "sample_shape_index_path must be inside tokenized_path so both artifacts publish atomically"
                        ) from error
                    if relative_index_path == Path("."):
                        raise ValueError("sample_shape_index_path must be a child of tokenized_path")

                    def prepare_assembly(assembly_path: Path) -> None:
                        write_sample_shape_index(
                            shape_index,
                            shape_summary,
                            assembly_path / relative_index_path,
                        )
                        write_rejected_samples(
                            preparation_rejections,
                            assembly_path / REJECTED_SAMPLES_FILE,
                        )

                    dataset_dict = finalize_resumable_dataset_dict(
                        dataset_dict,
                        data_args.tokenized_path,
                        prepare_assembly=prepare_assembly,
                    )
                    logger.info_rank0(f"Sample-shape index is saved at {data_args.sample_shape_index_path}.")
                elif resumable:
                    dataset_dict = finalize_resumable_dataset_dict(dataset_dict, data_args.tokenized_path)
                else:
                    dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

            return get_dataset_module(dataset_dict)
