# Copyright 2025 the LlamaFactory / SpeechLMM team.

from types import SimpleNamespace

import pytest

from llamafactory.model.model_utils.moe import config_is_qwen2_5_backbone, config_uses_qwen3_omni_moe_blocks


@pytest.mark.runs_on(["cpu", "mps"])
def test_native_qwen3_omni_uses_moe_blocks():
    config = SimpleNamespace(model_type="qwen3_omni_moe")
    assert config_uses_qwen3_omni_moe_blocks(config) is True


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_qwen3_uses_moe_blocks():
    config = SimpleNamespace(model_type="speechlmm", backbone_type="qwen3_omni", is_qwen2_5_backbone=False)
    assert config_uses_qwen3_omni_moe_blocks(config) is True


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_qwen25_skips_moe_blocks():
    config = SimpleNamespace(model_type="speechlmm", backbone_type="qwen2_5_omni", is_qwen2_5_backbone=True)
    assert config_is_qwen2_5_backbone(config) is True
    assert config_uses_qwen3_omni_moe_blocks(config) is False


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen25_omni_native_skips_moe_blocks():
    config = SimpleNamespace(model_type="qwen2_5_omni")
    assert config_is_qwen2_5_backbone(config) is True
    assert config_uses_qwen3_omni_moe_blocks(config) is False


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_qwen25_via_thinker_config():
    config = SimpleNamespace(model_type="speechlmm", thinker_config=SimpleNamespace(model_type="qwen2_5_omni_thinker"))
    assert config_is_qwen2_5_backbone(config) is True
    assert config_uses_qwen3_omni_moe_blocks(config) is False


@pytest.mark.runs_on(["cpu", "mps"])
def test_speechlmm_without_backbone_defaults_to_qwen3_moe():
    config = SimpleNamespace(model_type="speechlmm")
    assert config_is_qwen2_5_backbone(config) is False
    assert config_uses_qwen3_omni_moe_blocks(config) is True
