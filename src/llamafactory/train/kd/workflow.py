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

from typing import TYPE_CHECKING, Optional

from speechlmm.tokens import build_speechlmm_special_tokens

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..stateful_dataloader import build_stateful_data_config
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomKDTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_kd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    if finetuning_args.ref_model is None:
        raise ValueError("`stage: kd` needs `ref_model`, the path to the frozen teacher.")

    needs_lipread_tokens = model_args.use_speechlmm_wrapper or (
        data_args.template == "speechlmm"
        and (not finetuning_args.freeze_lipread_encoder or not finetuning_args.freeze_lipread_adapter)
    )
    special_tokens = model_args.add_special_tokens
    if needs_lipread_tokens:
        model_args.add_special_tokens = (special_tokens or []) + build_speechlmm_special_tokens(model_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # create_ref_model re-parses the copied ModelArguments, which expects a comma-separated string here.
    if isinstance(model_args.add_special_tokens, list):
        model_args.add_special_tokens = ",".join(model_args.add_special_tokens) or None
    ref_model = create_ref_model(model_args, finetuning_args)
    model_args.add_special_tokens = special_tokens

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        media_dir=data_args.media_dir,
        **tokenizer_module,
    )

    trainer = CustomKDTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        stateful_data_config=(
            build_stateful_data_config(data_args) if finetuning_args.use_stateful_dataloader else None
        ),
        **dataset_module,
        **tokenizer_module,
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
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"]]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
