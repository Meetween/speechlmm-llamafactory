# Copyright 2025 Meetween / SpeechLMM KD extension.
"""KD hyperparameters and extended ``FinetuningArguments`` (kept under ``train/kd`` only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...hparams.finetuning_args import FinetuningArguments as _BaseFinetuningArguments


@dataclass
class KDArguments:
    r"""Arguments for ``stage: kd`` (logit distillation; SpeechLMM KD decision docs)."""

    kd_ce_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for supervised CE on gold labels from the student forward."},
    )
    kd_jsd_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for mean token-wise symmetric JSD between student and teacher LM logits."},
    )
    kd_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature T applied to logits before softmax in the JSD term."},
    )
    kd_aut_mse_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for MSE between student and teacher AuT last hidden states (when available)."},
    )
    kd_enable_aut_mse: bool = field(
        default=True,
        metadata={
            "help": (
                "If set, add the AuT MSE term when the batch has ``input_features`` and both models implement "
                "``get_audio_features`` with matching hidden shapes; otherwise the term is skipped."
            )
        },
    )
    kd_stage0_projector_only: bool = field(
        default=False,
        metadata={
            "help": (
                "``stage: kd`` only: freeze all parameters except multimodal projectors "
                "(audio ``proj1``/``proj2``/``ln_post``; vision ``merger`` / ``merger_list``). "
                "Use ``finetuning_type: full`` for a strict Stage-0 warmup."
            )
        },
    )


@dataclass
class FinetuningArgumentsKD(KDArguments, _BaseFinetuningArguments):
    """Drop-in replacement for ``FinetuningArguments`` when training with ``stage: kd``."""

    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "kd"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.use_ref_model = (self.stage == "kd") or (
            self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]
        )
