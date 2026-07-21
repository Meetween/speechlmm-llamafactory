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

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

from speechlmm.data_loading_optimization.sample_shapes import (
    AUDIO_LAYOUTS_COLUMN,
    CONTEXT_LIMIT_COLUMN,
    IMAGE_LAYOUTS_COLUMN,
    PRETRUNCATE_TARGET_TOKENS_COLUMN,
    PRETRUNCATE_TOKENS_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
    VIDEO_LAYOUTS_COLUMN,
)

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..preparation_errors import (
    ALIGNMENT_ERROR_COLUMN,
    PROCESSING_ERROR_COLUMN,
    RecoverablePreparationError,
    build_rejection_record,
    deserialize_error,
    raise_recoverable_preparation_error,
    serialize_rejection,
)
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class DataExampleLengthStats:
    context_limit: int
    pretruncate_thinker_tokens: int
    pretruncate_valid_target_tokens: int


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _build_data_example_at_limit(
        self,
        input_ids: list[int],
        labels: list[int],
        encoded_pairs: list[tuple[list[int], list[int]]],
        cutoff_len: int,
    ) -> tuple[list[int], list[int]]:
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= cutoff_len:
                break

            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def _build_data_example(
        self,
        input_ids: list[int],
        labels: list[int],
        encoded_pairs: list[tuple[list[int], list[int]]],
    ) -> tuple[list[int], list[int]]:
        return self._build_data_example_at_limit(
            input_ids,
            labels,
            encoded_pairs,
            self.data_args.cutoff_len,
        )

    def _build_data_example_with_stats(
        self,
        input_ids: list[int],
        labels: list[int],
        encoded_pairs: list[tuple[list[int], list[int]]],
    ) -> tuple[list[int], list[int], DataExampleLengthStats]:
        full_limit = (
            len(input_ids)
            + sum(len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs)
            + (1 if self.template.efficient_eos else 0)
        )
        full_input_ids, full_labels = self._build_data_example_at_limit(
            list(input_ids),
            list(labels),
            encoded_pairs,
            max(1, full_limit),
        )
        limited_input_ids, limited_labels = self._build_data_example_at_limit(
            list(input_ids),
            list(labels),
            encoded_pairs,
            self.data_args.cutoff_len,
        )
        stats = DataExampleLengthStats(
            context_limit=self.data_args.cutoff_len,
            pretruncate_thinker_tokens=len(full_input_ids),
            pretruncate_valid_target_tokens=sum(token != IGNORE_INDEX for token in full_labels[1:]),
        )
        return limited_input_ids, limited_labels, stats

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        return self._build_data_example(input_ids, labels, encoded_pairs)

    def _encode_data_examples_batch(
        self, examples: dict[str, list[Any]]
    ) -> tuple[list[tuple[int, list[int], list[int], DataExampleLengthStats, Any]], dict[int, str]]:
        pending_examples = []
        rejected_samples: dict[int, str] = {}
        alignment_errors = examples.get(ALIGNMENT_ERROR_COLUMN)
        for index in range(len(examples["_prompt"])):
            try:
                if self.data_args.build_sample_shape_index and alignment_errors is not None:
                    alignment_error = alignment_errors[index]
                    if alignment_error is not None:
                        detail = deserialize_error(alignment_error)
                        raise RecoverablePreparationError(
                            detail["exception_message"],
                            processing_stage=detail["processing_stage"],
                            exception_class=detail["exception_class"],
                        )

                prompt = examples["_prompt"][index]
                response = examples["_response"][index]
                if not isinstance(prompt, list) or not isinstance(response, list):
                    raise RecoverablePreparationError(
                        "Prompt and response must both be message lists.",
                        processing_stage="message_validation",
                    )
                if len(prompt) % 2 != 1 or len(response) != 1:
                    if not self.data_args.build_sample_shape_index:
                        logger.warning_rank0(f"Dropped invalid example: {prompt + response}")
                        continue
                    raise RecoverablePreparationError(
                        "Supervised examples require an odd number of prompt messages and one response.",
                        processing_stage="message_validation",
                    )
                if any(
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                    for message in prompt + response
                ):
                    raise RecoverablePreparationError(
                        "Every message must contain string role and content fields.",
                        processing_stage="message_validation",
                    )

                images = examples["_images"][index] or []
                videos = examples["_videos"][index] or []
                audios = examples["_audios"][index] or []
                raw_messages = prompt + response
                try:
                    if self.data_args.build_sample_shape_index:
                        messages, layouts = self.template.mm_plugin.process_messages_with_layout(
                            raw_messages, images, videos, audios, self.processor
                        )
                    else:
                        messages = self.template.mm_plugin.process_messages(
                            raw_messages, images, videos, audios, self.processor
                        )
                        layouts = None
                    input_ids, labels = self.template.mm_plugin.process_token_ids(
                        [], [], images, videos, audios, self.tokenizer, self.processor
                    )
                except (OSError, ValueError) as error:
                    if not self.data_args.build_sample_shape_index:
                        raise
                    raise_recoverable_preparation_error(error, "multimodal_placeholder_expansion")
            except RecoverablePreparationError as error:
                if not self.data_args.build_sample_shape_index:
                    raise
                record = build_rejection_record(
                    examples,
                    index,
                    error,
                    source_column=SOURCE_ID_COLUMN,
                    sample_id_column=SAMPLE_ID_COLUMN,
                )
                rejected_samples[index] = serialize_rejection(record)
                continue
            pending_examples.append((index, messages, input_ids, labels, layouts))

        try:
            encoded_batch = self.template.encode_multiturn_batch(
                self.tokenizer,
                [item[1] for item in pending_examples],
                [examples["_system"][item[0]] for item in pending_examples],
                [examples["_tools"][item[0]] for item in pending_examples],
            )
        except (TypeError, ValueError) as batch_error:
            if not self.data_args.build_sample_shape_index:
                raise
            encoded_batch = []
            surviving_examples = []
            text_rejection_count = 0
            for item in pending_examples:
                index = item[0]
                try:
                    encoded = self.template.encode_multiturn(
                        self.tokenizer,
                        item[1],
                        examples["_system"][index],
                        examples["_tools"][index],
                    )
                except (TypeError, ValueError) as error:
                    recoverable = RecoverablePreparationError.from_exception(error, "text_tokenization")
                    rejected_samples[index] = serialize_rejection(
                        build_rejection_record(
                            examples,
                            index,
                            recoverable,
                            source_column=SOURCE_ID_COLUMN,
                            sample_id_column=SAMPLE_ID_COLUMN,
                        )
                    )
                    text_rejection_count += 1
                    continue
                surviving_examples.append(item)
                encoded_batch.append(encoded)
            pending_examples = surviving_examples
            if text_rejection_count == 0:
                raise batch_error

        encoded_examples = [
            (index, *self._build_data_example_with_stats(input_ids, labels, encoded_pairs), layouts)
            for (index, _messages, input_ids, labels, layouts), encoded_pairs in zip(
                pending_examples, encoded_batch, strict=True
            )
        ]
        return encoded_examples, rejected_samples

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        encoded_examples, rejected_samples = self._encode_data_examples_batch(examples)
        encoded_by_index = {item[0]: item[1:] for item in encoded_examples}
        output_indexes = (
            range(len(examples["_prompt"])) if self.data_args.build_sample_shape_index else encoded_by_index
        )
        for i in output_indexes:
            rejected = rejected_samples.get(i)
            if rejected is None:
                input_ids, labels, stats, layouts = encoded_by_index[i]
            else:
                input_ids, labels, layouts = [0], [IGNORE_INDEX], None
                stats = DataExampleLengthStats(self.data_args.cutoff_len, 0, 0)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            codec_tokens = examples.get("_codec_tokens", [None] * len(examples["_prompt"]))
            model_inputs["codec_tokens"].append(codec_tokens[i])
            if SOURCE_ID_COLUMN in examples:
                model_inputs[SOURCE_ID_COLUMN].append(examples[SOURCE_ID_COLUMN][i])
            if SAMPLE_ID_COLUMN in examples:
                model_inputs[SAMPLE_ID_COLUMN].append(examples[SAMPLE_ID_COLUMN][i])
            if self.data_args.build_sample_shape_index:
                model_inputs[PROCESSING_ERROR_COLUMN].append(rejected)
                model_inputs[AUDIO_LAYOUTS_COLUMN].append(
                    [] if layouts is None else [asdict(layout) for layout in layouts.audios]
                )
                model_inputs[IMAGE_LAYOUTS_COLUMN].append(
                    [] if layouts is None else [asdict(layout) for layout in layouts.images]
                )
                model_inputs[VIDEO_LAYOUTS_COLUMN].append(
                    [] if layouts is None else [asdict(layout) for layout in layouts.videos]
                )
                model_inputs[CONTEXT_LIMIT_COLUMN].append(stats.context_limit)
                model_inputs[PRETRUNCATE_TOKENS_COLUMN].append(stats.pretruncate_thinker_tokens)
                model_inputs[PRETRUNCATE_TARGET_TOKENS_COLUMN].append(stats.pretruncate_valid_target_tokens)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        encoded_examples, rejected_samples = self._encode_data_examples_batch(examples)
        if rejected_samples:
            raise RuntimeError("row quarantine is incompatible with packed preprocessing")
        for i, input_ids, labels, _stats, _layouts in encoded_examples:
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
