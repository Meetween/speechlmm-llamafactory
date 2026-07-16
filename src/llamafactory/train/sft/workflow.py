# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from speechlmm.data_loading_optimization.integration import (
    DynamicBatchingCheckpointCallback,
    build_dynamic_batch_plan,
    dynamic_plan_metrics,
)
from speechlmm.memory_estimation.probing import configure_memory_probe, get_memory_probe, record_memory

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    if getattr(model_args, "memory_probe", False):
        probe_dir = model_args.memory_probe_output_dir or f"{training_args.output_dir}/memory_probe"
        configure_memory_probe(
            enabled=True,
            output_dir=probe_dir,
            sync_cuda=model_args.memory_probe_sync_cuda,
            record_shapes=model_args.memory_probe_record_shapes,
            nvml=model_args.memory_probe_nvml,
            rank_zero_only=model_args.memory_probe_rank_zero_only,
            loss_breakdown=model_args.memory_probe_loss_breakdown,
            allocator_stats=model_args.memory_probe_allocator_stats,
            driver_memory=model_args.memory_probe_driver_memory,
            snapshot_rank=model_args.memory_probe_snapshot_rank,
            memory_history=model_args.memory_probe_memory_history,
            memory_history_max_entries=model_args.memory_probe_history_max_entries,
        )
        record_memory(
            "sft.run_start",
            extra={
                "output_dir": training_args.output_dir,
                "probe_dir": probe_dir,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "deepspeed": training_args.deepspeed,
            },
        )

    tokenizer_module = load_tokenizer(model_args)
    record_memory("sft.after_load_tokenizer")
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    record_memory("sft.after_get_template")
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    record_memory("sft.after_get_dataset")
    dynamic_batch_plan = None
    if data_args.dynamic_batching:
        dynamic_batch_plan = build_dynamic_batch_plan(
            train_dataset=dataset_module["train_dataset"],
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            processor=tokenizer_module.get("processor"),
        )
        callbacks = list(callbacks or [])
        callbacks.append(DynamicBatchingCheckpointCallback(dynamic_batch_plan))
        record_memory(
            "sft.after_dynamic_batch_plan",
            extra={
                "plan_hash": dynamic_batch_plan.plan_hash,
                "distributed_microsteps": len(dynamic_batch_plan.microsteps),
            },
        )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    record_memory("sft.after_load_model")
    recorder = get_memory_probe()
    if recorder is not None:
        recorder.start_memory_history()
        recorder.dump_snapshot("after_load_model")

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    record_memory("sft.after_data_collator")

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy(tokenizer=tokenizer)
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            dynamic_batch_plan=dynamic_batch_plan,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
    record_memory("sft.after_trainer_init")

    # Training
    if training_args.do_train:
        record_memory("sft.before_trainer_train")
        recorder = get_memory_probe()
        if recorder is not None:
            recorder.dump_snapshot("before_trainer_train")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        record_memory("sft.after_trainer_train")
        if not getattr(model_args, "memory_probe_skip_model_save", False):
            trainer.save_model()
            record_memory("sft.after_save_model")
        else:
            record_memory("sft.model_save_skipped")
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        if dynamic_batch_plan is not None:
            train_result.metrics.update(dynamic_plan_metrics(dynamic_batch_plan))

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                per_ds = [
                    [f"eval_{key}_loss", f"eval_{key}_accuracy", f"eval_{key}_wer", f"eval_{key}_cer"]
                    for key in dataset_module["eval_dataset"].keys()
                ]
                keys += sum(per_ds, [])
            else:
                keys += ["eval_loss", "eval_accuracy", "eval_wer", "eval_cer"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
    record_memory("sft.run_end")
    recorder = get_memory_probe()
    if recorder is not None:
        recorder.close()
