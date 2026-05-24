# Copyright 2025 Meetween / SpeechLMM KD extension.
"""Stage-0 projector-only freeze (``SpeechLMMv2_KD_Decision.docx``).

Freeze AuT/ViT backbone + LLM; leave **Qwen3-style projector weights** trainable:
audio ``proj1`` / ``proj2`` / ``ln_post``; vision ``merger`` / ``merger_list``.
Works with both flat ``audio_tower.*`` / ``visual.*`` names and nested ``thinker.*`` exports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _is_stage0_trainable_param_name(name: str) -> bool:
    """Return True if *name* should stay trainable during Stage-0 projector warmup."""
    if "patch_embed" in name:
        return False
    if "audio_tower" in name or "thinker.audio_tower" in name:
        return any(marker in name for marker in (".proj1.", ".proj2.", ".ln_post."))
    if ".visual." in name or name.startswith("visual."):
        return any(marker in name for marker in (".merger.", ".merger_list"))
    return False


def apply_kd_stage0_projector_only_freeze(model: torch.nn.Module) -> tuple[int, int, list[str]]:
    """Freeze all parameters, then unfreeze projector-related tensors only.

    Returns
    -------
    n_frozen, n_trainable, sample_trainable_names (up to 32, for logs).
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_names: list[str] = []
    for name, p in model.named_parameters():
        if _is_stage0_trainable_param_name(name):
            p.requires_grad = True
            trainable_names.append(name)

    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    return n_frozen, n_trainable, trainable_names[:32]
