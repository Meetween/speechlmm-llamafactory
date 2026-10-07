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

from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ..sft.trainer import CustomSeq2SeqTrainer
from .losses import jsd_loss


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class CustomKDTrainer(CustomSeq2SeqTrainer):
    r"""Seq2Seq trainer that adds a JSD term to a frozen teacher."""

    def __init__(self, ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None, **kwargs: Any) -> None:
        if ref_model is not None:
            disable_dropout_in_model(ref_model)

        super().__init__(**kwargs)
        self.ref_model = ref_model
        if ref_model is not None:
            ref_model.requires_grad_(False)
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, ref_model)
                else:
                    self.ref_model = prepare_fsdp(ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)

            self.ref_model.eval()
        self._kd_terms = {"train": [], "eval": []}

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.ref_model is None or self.finetuning_args.kd_jsd_weight <= 0:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        outputs = model(**inputs, use_cache=False, return_dict=True)
        teacher_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        # no_grad, not inference_mode: ZeRO-3 parameter hooks reject inference tensors.
        with torch.no_grad():
            teacher_logits = self.ref_model(**teacher_inputs, use_cache=False, return_dict=True).logits

        jsd = jsd_loss(
            outputs.logits, teacher_logits, inputs["labels"], self.finetuning_args.kd_temperature, IGNORE_INDEX
        )
        ce = outputs.loss.detach().float()
        bucket = "train" if model.training else "eval"
        self._kd_terms[bucket].append((ce.item(), jsd.detach().float().item()))
        loss = self.finetuning_args.kd_ce_weight * outputs.loss.float() + self.finetuning_args.kd_jsd_weight * jsd
        return (loss, outputs) if return_outputs else loss

    @override
    def evaluate(self, eval_dataset=None, *args, **kwargs):
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        metrics = super().evaluate(eval_dataset, *args, **kwargs)
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # Each split loss is a mean over its own rows. Weight by row count so a
        # short split does not count as much as a long one. The heal monitor has
        # 16 rows in every split, so this is their unweighted mean.
        if isinstance(dataset, dict):
            total = 0.0
            rows = 0
            for name, subset in dataset.items():
                loss = metrics.get(f"{metric_key_prefix}_{name}_loss")
                if loss is None:
                    continue
                n = len(subset)
                total += float(loss) * n
                rows += n
            if rows:
                metrics[f"{metric_key_prefix}_loss"] = total / rows
                self.log({f"{metric_key_prefix}_loss": metrics[f"{metric_key_prefix}_loss"]})
        return metrics

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Add the averaged CE and JSD terms to the training or eval log."""
        bucket = "eval" if "eval_loss" in logs else "train"
        terms = self._kd_terms[bucket]
        if terms and ("loss" in logs or "eval_loss" in logs):
            stacked = torch.tensor(terms, dtype=torch.float, device=self.accelerator.device).mean(dim=0)
            ce, jsd = self.accelerator.reduce(stacked, "mean").tolist()
            prefix = "eval_" if bucket == "eval" else ""
            logs[f"{prefix}kd_ce"] = ce
            logs[f"{prefix}kd_jsd"] = jsd
            terms.clear()
        return super().log(logs, *args, **kwargs)
