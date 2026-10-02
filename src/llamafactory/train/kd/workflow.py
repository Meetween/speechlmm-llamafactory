# Copyright 2025 Meetween / SpeechLMM KD extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Knowledge distillation. Reuses the SFT dataset and collator, plus a frozen teacher."""

from typing import TYPE_CHECKING, Optional

from speechlmm.tokens import build_speechlmm_special_tokens

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from ..stateful_dataloader import build_stateful_data_config
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .freeze import apply_kd_stage0_projector_only_freeze
from .trainer import CustomKDTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _load_teacher(model_args: "ModelArguments", finetuning_args: "FinetuningArguments"):
    """Load the frozen teacher.

    ``create_ref_model`` copies ``ModelArguments`` and re-runs ``__post_init__``,
    which splits ``add_special_tokens``. The SFT workflow has already turned that
    field into a list, so hand the copy a comma-joined string.
    """
    tokens = getattr(model_args, "add_special_tokens", None)
    if isinstance(tokens, (list, tuple)):
        joined = ",".join(str(token) for token in tokens if str(token).strip())
        model_args.add_special_tokens = joined or None
        try:
            return create_ref_model(model_args, finetuning_args)
        finally:
            model_args.add_special_tokens = tokens
    return create_ref_model(model_args, finetuning_args)


def run_kd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    if model_args.use_kt:
        raise NotImplementedError("`stage: kd` with KTransformers is not supported.")

    needs_lipread_tokens = model_args.use_speechlmm_wrapper or (
        data_args.template == "speechlmm"
        and (not finetuning_args.freeze_lipread_encoder or not finetuning_args.freeze_lipread_adapter)
    )
    if needs_lipread_tokens:
        original_add_special_tokens = getattr(model_args, "add_special_tokens", None) or []
        model_args.add_special_tokens = original_add_special_tokens + build_speechlmm_special_tokens(model_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if finetuning_args.kd_stage0_projector_only:
        n_frozen, n_train, sample = apply_kd_stage0_projector_only_freeze(model)
        logger.info_rank0(
            "KD projector-only freeze: trainable_param_tensors=%d frozen_param_tensors=%d sample_trainable=%s",
            n_train,
            n_frozen,
            sample,
        )
        if finetuning_args.finetuning_type == "lora":
            logger.warning_rank0_once(
                "kd_stage0_projector_only with finetuning_type=lora still trains LoRA adapters. "
                "Use finetuning_type=full for a projector-only run."
            )

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    stage0_ce_only = (
        finetuning_args.kd_stage0_projector_only
        and finetuning_args.kd_jsd_weight == 0
        and (not finetuning_args.kd_enable_aut_mse or finetuning_args.kd_aut_mse_weight == 0)
    )
    ref_model = None
    if finetuning_args.ref_model is not None and not stage0_ce_only:
        ref_model = _load_teacher(model_args, finetuning_args)
        logger.info_rank0("KD: loaded frozen teacher from `ref_model`.")
    elif training_args.do_train and not stage0_ce_only:
        raise ValueError("`ref_model` must point to the frozen teacher when training with `stage: kd`.")
    elif not stage0_ce_only:
        logger.warning_rank0("KD eval without `ref_model`: the JSD term is skipped.")

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        media_dir=data_args.media_dir,
        **tokenizer_module,
    )

    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy(tokenizer=tokenizer)
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            extra_ids = tokenizer.convert_tokens_to_ids([str(token) for token in extra_special_tokens])
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        gen_kwargs["eos_token_id"] = list(dict.fromkeys(all_eos_ids))
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    trainer = CustomKDTrainer(
        ref_model=ref_model,
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        stateful_data_config=(
            build_stateful_data_config(data_args) if finetuning_args.use_stateful_dataloader else None
        ),
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
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
        tokenizer.padding_side = "left"

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
