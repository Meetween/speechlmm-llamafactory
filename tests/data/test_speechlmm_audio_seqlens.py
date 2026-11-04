# Copyright 2025 the LlamaFactory / SpeechLMM team.

from types import SimpleNamespace

import pytest
import torch
from speechlmm.tokens import DUMMY_LIPREAD_FRAMES, LIPREAD_FRAME_SIZE

from llamafactory.data.collator import (
    _audio_seqlens_fallback,
    _dummy_lipread_batch,
    _feat_extract_output_length_fn,
    _invert_feat_extract_output_length,
    _is_speechlmm_model,
    _module_is_trainable,
    _qwen2_5_feat_extract_output_lengths,
    _reconcile_audio_placeholder_tokens,
)
from llamafactory.data.template import TEMPLATES
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


@pytest.mark.runs_on(["cpu", "mps"])
def test_reconcile_skips_dummy_audio_without_placeholders():
    ids = [1, 2, 3]
    feature = {"input_ids": ids[:], "attention_mask": [1] * 3, "labels": [IGNORE_INDEX] * 3}
    _reconcile_audio_placeholder_tokens(
        [feature],
        [1],
        {"feature_attention_mask": torch.ones(1, 1300)},
        _qwen25_config(audio_token_id=3),
    )
    assert feature["input_ids"] == ids


@pytest.mark.runs_on(["cpu", "mps"])
def test_reconcile_raises_on_truncated_audio_span():
    ids = [1, 10, 11, 11]
    feature = {"input_ids": ids[:], "attention_mask": [1] * 4, "labels": [IGNORE_INDEX] * 4}
    with pytest.raises(ValueError, match="cutoff_len"):
        _reconcile_audio_placeholder_tokens(
            [feature],
            [1],
            {"feature_attention_mask": torch.ones(1, 1300)},
            _qwen25_config(),
        )


@pytest.mark.runs_on(["cpu", "mps"])
def test_reconcile_skips_when_batch_has_no_audio_tokens():
    ids = [1, 2, 3]
    feature = {"input_ids": ids[:], "attention_mask": [1, 1, 1], "labels": [IGNORE_INDEX] * 3}
    _reconcile_audio_placeholder_tokens(
        [feature],
        [1],
        {"feature_attention_mask": torch.ones(1, 1300)},
        _qwen25_config(),
    )
    assert feature["input_ids"] == ids


@pytest.mark.runs_on(["cpu", "mps"])
def test_audio_seqlens_fallback_uses_feature_lengths():
    input_ids = torch.tensor([[10, 11, 11, 12]])
    seqlens = _audio_seqlens_fallback(
        input_ids,
        torch.ones(1, 1300),
        _qwen25_config(),
        is_speechlmm=False,
    )
    assert seqlens is not None
    assert seqlens.tolist() == [1300]


@pytest.mark.runs_on(["cpu", "mps"])
def test_audio_seqlens_fallback_skipped_for_speechlmm():
    input_ids = torch.tensor([[10, 11, 11, 12]])
    assert (
        _audio_seqlens_fallback(
            input_ids,
            torch.ones(1, 1300),
            _qwen25_config(),
            is_speechlmm=True,
        )
        is None
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_audio_seqlens_fallback_skipped_without_audio_tokens():
    input_ids = torch.tensor([[1, 2, 3]])
    assert (
        _audio_seqlens_fallback(
            input_ids,
            torch.ones(1, 1300),
            _qwen25_config(),
            is_speechlmm=False,
        )
        is None
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_audio_seqlens_fallback_skipped_without_feature_mask():
    input_ids = torch.tensor([[10, 11, 11, 12]])
    assert (
        _audio_seqlens_fallback(
            input_ids,
            None,
            _qwen25_config(),
            is_speechlmm=False,
        )
        is None
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_dummy_lipread_batch_matches_frame_constant():
    lipread, mask = _dummy_lipread_batch()
    assert lipread.shape == (1, DUMMY_LIPREAD_FRAMES, LIPREAD_FRAME_SIZE, LIPREAD_FRAME_SIZE)
    assert mask.shape == (1, DUMMY_LIPREAD_FRAMES, DUMMY_LIPREAD_FRAMES)
    assert mask.dtype == torch.uint8


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_model_and_trainable_helpers():
    assert _is_speechlmm_model(SimpleNamespace(config=SimpleNamespace(model_type="speechlmm"))) is True
    assert _is_speechlmm_model(SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_omni"))) is False
    assert _is_speechlmm_model(None) is False

    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(False)
    trainable = torch.nn.Linear(2, 2)
    assert _module_is_trainable(None) is False
    assert _module_is_trainable(frozen) is False
    assert _module_is_trainable(trainable) is True


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_qwen2_omni_template_uses_qwen25_tokens():
    plugin = TEMPLATES["speechlmm_qwen2_omni"].mm_plugin
    assert plugin.image_token == "<|IMAGE|>"
    assert plugin.video_token == "<|VIDEO|>"
    assert plugin.audio_token == "<|AUDIO|>"
    assert plugin.audio_bos_token == "<|audio_bos|>"
    assert plugin.audio_eos_token == "<|audio_eos|>"
