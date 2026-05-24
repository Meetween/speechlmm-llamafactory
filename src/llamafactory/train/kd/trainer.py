# Copyright 2025 Meetween / SpeechLMM KD extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# KD trainer: dual forward (student + frozen ``ref_model`` teacher) with
# ``w_ce * CE + w_jsd * JSD + w_aut * MSE(AuT)`` per SpeechLMM KD decision docs.

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..sft.trainer import CustomSeq2SeqTrainer
from .losses import kd_aut_mse_from_audio_features, kd_ce_from_logits, kd_jsd_shifted


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


def _tensor_device(inputs: Dict[str, Any]) -> torch.device:
    for v in inputs.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")


def _fp32_logits_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return nullcontext()


class CustomKDTrainer(CustomSeq2SeqTrainer):
    """Seq2Seq trainer with frozen **teacher** ``ref_model`` and KD composite loss."""

    def __init__(self, ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None, **kwargs: Any) -> None:
        if ref_model is not None:
            disable_dropout_in_model(ref_model)
        self.ref_model = ref_model
        super().__init__(**kwargs)
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
            # Match DPO: ZeRO-3/FSDP must wrap ref_model or embedding weights are invalid (2-D error).
            if self.is_deepspeed_enabled:
                if not (
                    getattr(self.ref_model, "is_loaded_in_8bit", False)
                    or getattr(self.ref_model, "is_loaded_in_4bit", False)
                ):
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()
        self._kd_jsd_shape_warned = False
        self._kd_aut_shape_logged = False

    def _needs_teacher_forward(self) -> bool:
        fa = self.finetuning_args
        if self.ref_model is None:
            return False
        if fa.kd_jsd_weight and fa.kd_jsd_weight > 0:
            return True
        if fa.kd_enable_aut_mse and fa.kd_aut_mse_weight and fa.kd_aut_mse_weight > 0:
            return True
        return False

    def _unwrap(self, mod: torch.nn.Module) -> torch.nn.Module:
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.unwrap_model(mod)
        return mod

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.ref_model is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        fa = self.finetuning_args
        device = _tensor_device(inputs)

        outputs = model(**inputs, use_cache=False, return_dict=True)
        loss_ce = outputs.loss
        logits_s = outputs.logits
        if loss_ce is None:
            loss_ce = kd_ce_from_logits(logits_s, labels, IGNORE_INDEX)

        logits_s_f = logits_s.float()
        jsd = torch.zeros((), device=logits_s_f.device, dtype=logits_s_f.dtype)

        if self._needs_teacher_forward():
            with torch.inference_mode():
                with _fp32_logits_ctx(device):
                    out_t = self.ref_model(**inputs, use_cache=False, return_dict=True)
            logits_t_f = out_t.logits.float()

            if logits_s_f.shape == logits_t_f.shape:
                jsd = kd_jsd_shifted(logits_s_f, logits_t_f, labels, fa.kd_temperature, IGNORE_INDEX)
            elif not self._kd_jsd_shape_warned:
                logger.warning_rank0_once(
                    "KD: student and teacher logits shapes differ; JSD term is skipped (CE + optional AuT only)."
                )
                self._kd_jsd_shape_warned = True

        loss = fa.kd_ce_weight * loss_ce + fa.kd_jsd_weight * jsd

        if (
            self._needs_teacher_forward()
            and fa.kd_enable_aut_mse
            and fa.kd_aut_mse_weight > 0
            and inputs.get("input_features") is not None
        ):
            aut = kd_aut_mse_from_audio_features(
                self._unwrap(model),
                self._unwrap(self.ref_model),
                inputs,
            )
            if aut is not None:
                loss = loss + fa.kd_aut_mse_weight * aut
            elif not self._kd_aut_shape_logged and inputs.get("input_features") is not None:
                logger.warning_rank0_once(
                    "KD: AuT MSE skipped (missing ``get_audio_features`` or mismatched hidden shapes). "
                    "Set ``kd_enable_aut_mse: false`` to silence this when expected."
                )
                self._kd_aut_shape_logged = True

        return (loss, outputs) if return_outputs else loss
