# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import re
from collections import Counter
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from speechlmm.data_loading_optimization import (
    DynamicBatchPlan,
    DynamicEvaluationDataset,
    DynamicEvaluationPlan,
    DynamicTrainingDataset,
    GlobalDynamicBatchSampler,
    count_valid_shifted_target_tokens,
)
from speechlmm.data_loading_optimization.integration import resume_start_microstep
from speechlmm.memory_estimation.probing import get_memory_probe, record_memory
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import seed_worker
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _probe_tensors(inputs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {key: inputs[key] for key in keys if key in inputs and torch.is_tensor(inputs[key])}


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        dynamic_batch_plan: Optional[DynamicBatchPlan] = None,
        dynamic_evaluation_plans: Optional[dict[str, DynamicEvaluationPlan]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        self.dynamic_batch_plan = dynamic_batch_plan
        self.dynamic_evaluation_plans = dict(dynamic_evaluation_plans or {})
        if dynamic_batch_plan is not None and kwargs.get("train_dataset") is not None:
            kwargs["train_dataset"] = DynamicTrainingDataset(kwargs["train_dataset"])
        eval_dataset = kwargs.get("eval_dataset")
        if self.dynamic_evaluation_plans:
            if set(self.dynamic_evaluation_plans) == {"global"}:
                kwargs["eval_dataset"] = DynamicEvaluationDataset(
                    eval_dataset,
                    self.dynamic_evaluation_plans["global"],
                )
            elif isinstance(eval_dataset, dict):
                missing = set(eval_dataset) - set(self.dynamic_evaluation_plans)
                if missing:
                    raise ValueError(f"dynamic evaluation plans are missing datasets: {sorted(missing)}")
                kwargs["eval_dataset"] = {
                    name: DynamicEvaluationDataset(dataset, self.dynamic_evaluation_plans[name])
                    for name, dataset in eval_dataset.items()
                }
            elif eval_dataset is not None:
                if set(self.dynamic_evaluation_plans) != {"validation"}:
                    raise ValueError("single validation dataset requires a plan named 'validation'")
                kwargs["eval_dataset"] = DynamicEvaluationDataset(
                    eval_dataset, self.dynamic_evaluation_plans["validation"]
                )
        if dynamic_batch_plan is not None:
            training_args.accelerator_config.split_batches = False
            training_args.accelerator_config.dispatch_batches = False
            training_args.accelerator_config.even_batches = False
            training_args.average_tokens_across_devices = True
            # Sampler owns resume offsets; disable Trainer's generic batch skip.
            training_args.ignore_data_skip = True
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        self._last_dynamic_metrics_step = 0
        self._dynamic_eval_loss_numerator = 0.0
        self._dynamic_eval_valid_tokens = 0
        self._dynamic_eval_dataset_numerators: dict[str, float] = {}
        self._dynamic_eval_dataset_tokens: dict[str, int] = {}
        self._dynamic_eval_microstep = 0
        self._active_dynamic_evaluation_plan: Optional[DynamicEvaluationPlan] = None
        if processor is not None and dynamic_batch_plan is None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False
        elif dynamic_batch_plan is not None:
            self.model_accepts_loss_kwargs = True

        # find_labels() auto-detects all *label* params (e.g. codec_labels),
        # but only "labels" is guaranteed present in every batch.
        self.label_names = ["labels"]

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_train_dataloader(self) -> "torch.utils.data.DataLoader":
        if self.dynamic_batch_plan is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataset = self.train_dataset
        data_collator = self.data_collator
        try:
            import datasets

            if isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description="Training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="Training")
        except ImportError:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="Training")

        start_microstep = 0
        if self.args.resume_from_checkpoint:
            start_microstep = resume_start_microstep(
                plan=self.dynamic_batch_plan,
                resume=self.args.resume_from_checkpoint,
                output_dir=self.args.output_dir,
                global_step=int(getattr(self.state, "global_step", 0) or 0),
            )

        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader_params = {
            "batch_sampler": GlobalDynamicBatchSampler(self.dynamic_batch_plan, start_microstep=start_microstep),
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)

    @override
    def get_eval_dataloader(
        self, eval_dataset: Optional["torch.utils.data.Dataset"] = None
    ) -> "torch.utils.data.DataLoader":
        if self.dynamic_batch_plan is None:
            return super().get_eval_dataloader(eval_dataset)
        if isinstance(eval_dataset, str):
            if not isinstance(self.eval_dataset, dict) or eval_dataset not in self.eval_dataset:
                raise ValueError(f"unknown dynamic evaluation dataset: {eval_dataset!r}")
            dataset = self.eval_dataset[eval_dataset]
        else:
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(dataset, DynamicEvaluationDataset):
            raise ValueError("dynamic batching evaluation requires a frozen DynamicEvaluationDataset")
        data_collator = self._get_collator_with_removed_columns(self.data_collator, description="Evaluation")
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader_params = {
            "batch_sampler": GlobalDynamicBatchSampler(dataset.dynamic_evaluation_plan),
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)

    @override
    def _get_num_items_in_batch(self, batch_samples, device):
        if self.dynamic_batch_plan is None:
            return super()._get_num_items_in_batch(batch_samples, device)
        if not batch_samples or "labels" not in batch_samples[0]:
            raise ValueError("dynamic batching requires labels in every microbatch")
        num_items = count_valid_shifted_target_tokens(batch_samples, ignore_index=IGNORE_INDEX)
        if self.args.world_size > 1:
            num_items = self.accelerator.gather(num_items.to(device)).sum()
        return num_items.to(device)

    def _dynamic_step_metrics(self, global_step: int) -> dict[str, float]:
        if self.dynamic_batch_plan is None or global_step <= 0:
            return {}
        accumulation = self.dynamic_batch_plan.settings.gradient_accumulation_steps
        start = (global_step - 1) * accumulation
        steps = self.dynamic_batch_plan.microsteps[start : start + accumulation]
        if not steps:
            return {}
        batches = [batch for step in steps for batch in step.local_batches]
        local = [step.local_batches[self.args.process_index] for step in steps]
        padded = sum(batch.estimate.batch_size * batch.estimate.padded_thinker_tokens for batch in batches)
        useful = sum(batch.unpadded_thinker_tokens for batch in batches)
        straggler_ratio = max(
            max(batch.estimate.thinker_activation_bytes for batch in step.local_batches)
            / min(batch.estimate.thinker_activation_bytes for batch in step.local_batches)
            for step in steps
        )
        utilization = max(batch.estimate.allocated_peak_bytes / batch.budget_bytes for batch in batches)
        allocator_slack = 0
        measured_allocated = 0
        measured_reserved = 0
        if torch.cuda.is_available():
            local_allocated = int(torch.cuda.max_memory_allocated())
            local_reserved = int(torch.cuda.max_memory_reserved())
            peak_values = torch.tensor(
                [
                    local_allocated,
                    local_reserved,
                    max(0, local_reserved - local_allocated),
                ],
                dtype=torch.int64,
                device=self.accelerator.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(peak_values, op=torch.distributed.ReduceOp.MAX)
            measured_allocated, measured_reserved, allocator_slack = (int(value) for value in peak_values.tolist())
        predicted_peak = max(batch.estimate.allocated_peak_bytes for batch in batches)
        metrics = {
            "dynamic_local_samples": float(sum(index >= 0 for batch in local for index in batch.dataset_indices)),
            "dynamic_global_samples": float(sum(index >= 0 for batch in batches for index in batch.dataset_indices)),
            "dynamic_synchronization_samples": float(
                sum(index < 0 for batch in batches for index in batch.dataset_indices)
            ),
            "dynamic_local_valid_target_tokens": float(sum(batch.estimate.valid_target_tokens for batch in local)),
            "dynamic_valid_target_tokens": float(sum(batch.estimate.valid_target_tokens for batch in batches)),
            "dynamic_thinker_tokens": float(useful),
            "dynamic_batch_size_min": float(min(len(batch.dataset_indices) for batch in batches)),
            "dynamic_batch_size_max": float(max(len(batch.dataset_indices) for batch in batches)),
            "dynamic_audio_feature_frames": float(sum(batch.audio_feature_frames for batch in batches)),
            "dynamic_audio_chunks": float(sum(step.global_audio_chunks for step in steps)),
            "dynamic_visual_grids": float(sum(batch.estimate.visual_grid_count for batch in batches)),
            "dynamic_visual_patch_tokens": float(sum(batch.visual_patch_tokens for batch in batches)),
            "dynamic_lipread_frames": float(sum(batch.lipread_frames for batch in batches)),
            "dynamic_predicted_peak_bytes": float(predicted_peak),
            "gpu_peak_allocated_bytes": float(measured_allocated),
            "gpu_peak_reserved_bytes": float(measured_reserved),
            "dynamic_allocated_prediction_error_bytes": float(measured_allocated - predicted_peak),
            "dynamic_allocated_prediction_ratio": (measured_allocated / predicted_peak if predicted_peak else 0.0),
            "dynamic_reserved_prediction_error_bytes": float(measured_reserved - predicted_peak),
            "dynamic_reserved_prediction_ratio": (measured_reserved / predicted_peak if predicted_peak else 0.0),
            "dynamic_target_utilization": utilization,
            "dynamic_target_unused_fraction": 1.0 - self.dynamic_batch_plan.settings.target_memory_used,
            "dynamic_allocator_reserved_unused_bytes": float(allocator_slack),
            "dynamic_padding_efficiency": useful / padded,
            "dynamic_predicted_rank_straggler_ratio": straggler_ratio,
        }
        source_counts = Counter(batch.source_id for batch in batches)
        requested = self.dynamic_batch_plan.settings.source_probabilities or {}
        for source in sorted(set(source_counts) | set(requested)):
            safe_source = re.sub(r"[^A-Za-z0-9_]+", "_", source).strip("_") or "source"
            metrics[f"dynamic_source_{safe_source}_realized_fraction"] = source_counts[source] / len(batches)
            if source in requested:
                metrics[f"dynamic_source_{safe_source}_requested_probability"] = float(requested[source])
        bottlenecks = Counter(batch.estimate.bottleneck_phase.split(":", 1)[0] for batch in batches)
        for phase, count in bottlenecks.items():
            metrics[f"dynamic_bottleneck_{phase}_fraction"] = count / len(batches)
        return metrics

    @override
    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ) -> None:
        current_step = int(self.state.global_step)
        if self.dynamic_batch_plan is not None and current_step > self._last_dynamic_metrics_step:
            # CallbackHandler.on_log clears control.should_log. Preserve the
            # Trainer decision across this additional metrics event so the
            # base implementation still publishes its normal per-step loss,
            # gradient norm, and learning-rate record.
            should_log = self.control.should_log
            self.log(self._dynamic_step_metrics(current_step), start_time)
            self.control.should_log = should_log
            self._last_dynamic_metrics_step = current_step
        return super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        recorder = get_memory_probe()
        if recorder is not None:
            recorder.set_step(getattr(self.state, "global_step", None))
            recorder.dump_zero3_execution_trace(model)

        record_memory(
            "trainer.compute_loss.before",
            tensors=_probe_tensors(
                inputs,
                (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "input_features",
                    "feature_attention_mask",
                    "position_ids",
                    "codec_labels",
                ),
            ),
        )
        try:
            loss = super().compute_loss(model, inputs, *args, **kwargs)
        except Exception as exc:
            if recorder is not None:
                recorder.record_exception("trainer.compute_loss.exception", exc)
            raise

        tensors = {"loss": loss[0] if isinstance(loss, tuple) and torch.is_tensor(loss[0]) else loss}
        record_memory("trainer.compute_loss.after", tensors={k: v for k, v in tensors.items() if torch.is_tensor(v)})
        return loss

    @override
    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        record_memory(
            "trainer.prepare_inputs.before",
            tensors=_probe_tensors(inputs, ("input_ids", "attention_mask", "labels")),
        )
        try:
            prepared = super()._prepare_inputs(inputs)
        except Exception as exc:
            recorder = get_memory_probe()
            if recorder is not None:
                recorder.record_exception("trainer.prepare_inputs.exception", exc)
            raise

        record_memory(
            "trainer.prepare_inputs.after",
            tensors=_probe_tensors(
                prepared,
                (
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "input_features",
                    "feature_attention_mask",
                    "position_ids",
                    "codec_labels",
                ),
            ),
        )
        return prepared

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        recorder = get_memory_probe()
        if recorder is not None:
            recorder.set_step(getattr(self.state, "global_step", None))

        record_memory("trainer.training_step.before")
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except Exception as exc:
            if recorder is not None:
                recorder.record_exception("trainer.training_step.exception", exc)
            raise

        record_memory("trainer.training_step.after_backward", tensors={"detached_loss": loss})
        if recorder is not None:
            recorder.dump_snapshot("after_backward")
        return loss

    @override
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        resolved = self.eval_dataset if eval_dataset is None else eval_dataset
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )
        if not self.dynamic_evaluation_plans or not isinstance(resolved, dict):
            return metrics

        numerators = []
        token_counts = []
        losses = []
        for name in resolved:
            prefix = f"{metric_key_prefix}_{name}"
            numerator_key = f"{prefix}_loss_sum"
            tokens_key = f"{prefix}_valid_target_tokens"
            loss_key = f"{prefix}_loss"
            if numerator_key not in metrics or tokens_key not in metrics or loss_key not in metrics:
                raise ValueError(f"dynamic evaluation metrics are incomplete for {name!r}")
            numerators.append(float(metrics[numerator_key]))
            token_counts.append(float(metrics[tokens_key]))
            losses.append(float(metrics[loss_key]))
        aggregate = {
            f"{metric_key_prefix}_global_loss": sum(numerators) / sum(token_counts),
            f"{metric_key_prefix}_macro_dataset_loss": sum(losses) / len(losses),
            f"{metric_key_prefix}_global_valid_target_tokens": sum(token_counts),
        }
        metrics.update(aggregate)
        self.log(aggregate)
        return metrics

    @override
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        dataset = getattr(dataloader, "dataset", None)
        plan = getattr(dataset, "dynamic_evaluation_plan", None)
        if plan is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        self._active_dynamic_evaluation_plan = plan
        self._dynamic_eval_loss_numerator = 0.0
        self._dynamic_eval_valid_tokens = 0
        self._dynamic_eval_dataset_numerators = dict.fromkeys(plan.dataset_names, 0.0)
        self._dynamic_eval_dataset_tokens = dict.fromkeys(plan.dataset_names, 0)
        self._dynamic_eval_microstep = 0
        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._active_dynamic_evaluation_plan = None

        if self._dynamic_eval_valid_tokens <= 0:
            raise ValueError(f"validation dataset {plan.dataset_name!r} has no valid shifted target tokens")
        if self._dynamic_eval_microstep != len(plan.microsteps):
            raise RuntimeError("dynamic evaluation did not execute every planned distributed microstep")
        exact_loss = self._dynamic_eval_loss_numerator / self._dynamic_eval_valid_tokens
        metrics = dict(output.metrics)
        metrics[f"{metric_key_prefix}_loss"] = exact_loss
        metrics[f"{metric_key_prefix}_global_loss"] = exact_loss
        metrics[f"{metric_key_prefix}_loss_sum"] = self._dynamic_eval_loss_numerator
        metrics[f"{metric_key_prefix}_valid_target_tokens"] = float(self._dynamic_eval_valid_tokens)
        dataset_losses = []
        for dataset_name in plan.dataset_names:
            tokens = self._dynamic_eval_dataset_tokens[dataset_name]
            if tokens <= 0:
                raise ValueError(f"validation dataset {dataset_name!r} has no valid shifted target tokens")
            numerator = self._dynamic_eval_dataset_numerators[dataset_name]
            dataset_loss = numerator / tokens
            dataset_losses.append(dataset_loss)
            dataset_prefix = f"{metric_key_prefix}_{dataset_name}"
            metrics[f"{dataset_prefix}_loss"] = dataset_loss
            metrics[f"{dataset_prefix}_loss_sum"] = numerator
            metrics[f"{dataset_prefix}_valid_target_tokens"] = float(tokens)
            metrics[f"{dataset_prefix}_selected_samples"] = float(plan.selected_count_for_dataset(dataset_name))
            metrics[f"{dataset_prefix}_rejected_samples"] = float(plan.rejected_count_for_dataset(dataset_name))
        metrics[f"{metric_key_prefix}_macro_dataset_loss"] = sum(dataset_losses) / len(dataset_losses)
        metrics[f"{metric_key_prefix}_selected_samples"] = float(plan.selected_count)
        metrics[f"{metric_key_prefix}_rejected_samples"] = float(len(plan.skipped_samples))
        metrics[f"{metric_key_prefix}_distributed_microsteps"] = float(len(plan.microsteps))
        return output._replace(metrics=metrics, num_samples=plan.selected_count)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        local_valid_tokens = None
        if self._active_dynamic_evaluation_plan is not None:
            if labels is None:
                raise ValueError("dynamic evaluation requires labels")
            local_valid_tokens = (labels[..., 1:] != IGNORE_INDEX).sum().to(self.accelerator.device)

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if self._active_dynamic_evaluation_plan is not None:
            if loss is None or local_valid_tokens is None:
                raise ValueError("dynamic evaluation requires a scalar loss")
            plan = self._active_dynamic_evaluation_plan
            if self._dynamic_eval_microstep >= len(plan.microsteps):
                raise RuntimeError("dynamic evaluation executed more steps than its plan")
            local_batch = plan.microsteps[self._dynamic_eval_microstep].local_batches[self.args.process_index]
            dataset_names = plan.dataset_names
            if local_batch.source_id not in dataset_names:
                raise RuntimeError(f"dynamic evaluation batch has unknown source {local_batch.source_id!r}")
            global_valid_tokens = local_valid_tokens.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_valid_tokens, op=torch.distributed.ReduceOp.SUM)
                world_size = torch.distributed.get_world_size()
            if int(local_valid_tokens.item()) == 0:
                # Synchronization duplicates deliberately mask every label.
                # Their local mean loss can be NaN (0 / 0), but their exact
                # token-weighted contribution is zero.
                local_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
            else:
                local_numerator = (
                    loss.detach().double().to(self.accelerator.device) * global_valid_tokens.double() / world_size
                )
                if not torch.isfinite(local_numerator):
                    raise FloatingPointError(
                        "non-finite validation loss on a rank with valid target tokens: "
                        f"dataset={local_batch.source_id!r}, "
                        f"rank={self.args.process_index}, "
                        f"valid_tokens={int(local_valid_tokens.item())}"
                    )
            dataset_values = torch.zeros(
                2 * len(dataset_names),
                dtype=torch.float64,
                device=self.accelerator.device,
            )
            dataset_index = dataset_names.index(local_batch.source_id)
            dataset_values[dataset_index] = local_numerator
            dataset_values[len(dataset_names) + dataset_index] = local_valid_tokens.double()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(dataset_values, op=torch.distributed.ReduceOp.SUM)
            for index, dataset_name in enumerate(dataset_names):
                numerator = float(dataset_values[index].item())
                tokens = int(dataset_values[len(dataset_names) + index].item())
                self._dynamic_eval_dataset_numerators[dataset_name] += numerator
                self._dynamic_eval_dataset_tokens[dataset_name] += tokens
                self._dynamic_eval_loss_numerator += numerator
                self._dynamic_eval_valid_tokens += tokens
            self._dynamic_eval_microstep += 1
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        # Dynamic validation computes its exact token-weighted loss above. In
        # loss-only evaluation, returning the rank-local labels would make the
        # base Trainer all-gather unequal leading dimensions because dynamic
        # batches can contain different sample counts on different ranks.
        if self._active_dynamic_evaluation_plan is not None and prediction_loss_only:
            labels = None

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
