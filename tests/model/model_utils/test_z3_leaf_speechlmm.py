from types import SimpleNamespace

from llamafactory.model.model_utils.moe import config_uses_qwen3_omni_moe_blocks


def test_native_qwen3_omni_uses_moe_blocks():
    config = SimpleNamespace(model_type="qwen3_omni_moe")
    assert config_uses_qwen3_omni_moe_blocks(config) is True


def test_speechlmm_qwen3_uses_moe_blocks():
    config = SimpleNamespace(model_type="speechlmm", backbone_type="qwen3_omni", is_qwen2_5_backbone=False)
    assert config_uses_qwen3_omni_moe_blocks(config) is True


def test_speechlmm_qwen25_skips_moe_blocks():
    config = SimpleNamespace(model_type="speechlmm", backbone_type="qwen2_5_omni", is_qwen2_5_backbone=True)
    assert config_uses_qwen3_omni_moe_blocks(config) is False


def test_qwen25_omni_native_skips_moe_blocks():
    config = SimpleNamespace(model_type="qwen2_5_omni")
    assert config_uses_qwen3_omni_moe_blocks(config) is False
