# Copyright 2025 the LlamaFactory / SpeechLMM team.

from types import SimpleNamespace

import pytest
import torch

from llamafactory.data.collator import (
    _feat_extract_output_length_fn,
    _invert_feat_extract_output_length,
    _qwen2_5_feat_extract_output_lengths,
    _reconcile_audio_placeholder_tokens,
)
from llamafactory.extras.constants import IGNORE_INDEX


def _qwen25_config(**extra):
    values = {
        "model_type": "speechlmm",
        "backbone_type": "qwen2_5_omni",
        "is_qwen2_5_backbone": True,
        "audio_start_token_id": 10,
        "audio_end_token_id": 12,
        "audio_token_id": 11,
    }
    values.update(extra)
    return SimpleNamespace(**values)


def _rope_audio_len(raw_length: int) -> int:
    return ((raw_length - 1) // 2 + 1 - 2) // 2 + 1


@pytest.mark.runs_on(["cpu", "mps"])
def test_feat_extract_fn_uses_qwen25_formula():
    fn = _feat_extract_output_length_fn(_qwen25_config())
    assert fn is _qwen2_5_feat_extract_output_lengths
    assert int(fn(1300)) == _rope_audio_len(1300) == 325


@pytest.mark.runs_on(["cpu", "mps"])
def test_feat_extract_fn_uses_qwen25_via_thinker_config():
    config = SimpleNamespace(model_type="speechlmm", thinker_config=SimpleNamespace(model_type="qwen2_5_omni_thinker"))
    assert _feat_extract_output_length_fn(config) is _qwen2_5_feat_extract_output_lengths


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen25_invert_roundtrips_token_count():
    fn = _qwen2_5_feat_extract_output_lengths
    raw = _invert_feat_extract_output_length(324, fn)
    assert int(fn(raw)) == 324
    assert _rope_audio_len(raw) == 324


@pytest.mark.runs_on(["cpu", "mps"])
def test_reconcile_shrinks_audio_placeholders():
    mask_len = 1300
    expected = int(_qwen2_5_feat_extract_output_lengths(mask_len))
    too_many = expected + 4
    ids = [1, 10] + [11] * too_many + [12, 2]
    feature = {"input_ids": ids[:], "attention_mask": [1] * len(ids), "labels": [IGNORE_INDEX] * len(ids)}
    _reconcile_audio_placeholder_tokens(
        [feature],
        [1],
        {"feature_attention_mask": torch.ones(1, mask_len)},
        _qwen25_config(),
    )
    assert feature["input_ids"].count(11) == expected
    assert len(feature["input_ids"]) == len(feature["attention_mask"]) == len(feature["labels"])


@pytest.mark.runs_on(["cpu", "mps"])
def test_reconcile_expands_audio_placeholders():
    mask_len = 1300
    expected = int(_qwen2_5_feat_extract_output_lengths(mask_len))
    too_few = expected - 4
    ids = [1, 10] + [11] * too_few + [12, 2]
    feature = {"input_ids": ids[:], "attention_mask": [1] * len(ids), "labels": [IGNORE_INDEX] * len(ids)}
    _reconcile_audio_placeholder_tokens(
        [feature],
        [1],
        {"feature_attention_mask": torch.ones(1, mask_len)},
        _qwen25_config(),
    )
    assert feature["input_ids"].count(11) == expected
    assert feature["labels"][feature["input_ids"].index(12) - 1] == IGNORE_INDEX
