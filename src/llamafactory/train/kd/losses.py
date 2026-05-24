# Copyright 2025 Meetween / SpeechLMM KD extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""KD loss pieces aligned with SpeechLMM KD decision docs (CE + symmetric JSD + optional AuT MSE)."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def kd_jsd_shifted(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    ignore_index: int,
) -> torch.Tensor:
    r"""Mean symmetric JSD over **next-token** positions (shifted vs ``labels``).

    Uses mixture ``m = 0.5 (p_s + p_t)`` with ``p_* = softmax(logits_* / T)``.
    """
    if temperature <= 0:
        raise ValueError("kd_temperature must be positive.")

    logits_s = logits_s[:, :-1, :].contiguous()
    logits_t = logits_t[:, :-1, :].contiguous()
    lab = labels[:, 1:].contiguous()
    mask = lab != ignore_index
    if not mask.any():
        return logits_s.sum() * 0.0

    ls = logits_s / temperature
    lt = logits_t / temperature
    log_p_s = F.log_softmax(ls, dim=-1)
    log_p_t = F.log_softmax(lt, dim=-1)
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()
    m = (0.5 * (p_s + p_t)).clamp(min=1e-8)
    log_m = m.log()
    jsd_tok = 0.5 * ((p_s * (log_p_s - log_m)).sum(dim=-1) + (p_t * (log_p_t - log_m)).sum(dim=-1))
    jsd_tok = jsd_tok * mask.to(jsd_tok.dtype)
    denom = mask.sum().to(jsd_tok.dtype).clamp(min=1.0)
    return jsd_tok.sum() / denom


def kd_ce_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Standard causal LM CE when ``outputs.loss`` is unavailable."""
    logits = logits[:, :-1, :].contiguous()
    lab = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        lab.view(-1),
        ignore_index=ignore_index,
    )


def kd_aut_mse_from_audio_features(
    unwrap_student: Any,
    unwrap_teacher: Any,
    inputs: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """MSE on ``audio_tower`` last hidden if both models expose ``get_audio_features`` and shapes match."""
    feats = inputs.get("input_features")
    if feats is None:
        return None
    if not hasattr(unwrap_student, "get_audio_features") or not hasattr(unwrap_teacher, "get_audio_features"):
        return None

    kwargs: Dict[str, Any] = {
        "feature_attention_mask": inputs.get("feature_attention_mask"),
        "audio_feature_lengths": inputs.get("audio_feature_lengths"),
    }
    with torch.no_grad():
        h_t = unwrap_teacher.get_audio_features(feats, **kwargs).last_hidden_state.float()
    h_s = unwrap_student.get_audio_features(feats, **kwargs).last_hidden_state.float()
    if h_s.shape != h_t.shape:
        return None
    return F.mse_loss(h_s, h_t)
