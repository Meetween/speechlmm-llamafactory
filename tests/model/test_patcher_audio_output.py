# Copyright 2025 the LlamaFactory team.
from types import SimpleNamespace

import pytest

import llamafactory.model.patcher as patcher_mod


def _stub_patch_config_helpers(monkeypatch) -> None:
    monkeypatch.setattr(patcher_mod, "infer_optim_dtype", lambda **kwargs: None)
    for name in (
        "configure_attn_implementation",
        "configure_rope",
        "configure_longlora",
        "configure_quantization",
        "configure_moe",
        "configure_visual_model",
        "configure_packing",
        "configure_kv_cache",
        "patch_qwen3_omni_moe_thinker_text_sparse_moe_block",
    ):
        monkeypatch.setattr(patcher_mod, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(patcher_mod, "is_deepspeed_zero3_enabled", lambda: False)
    monkeypatch.setattr(patcher_mod, "is_fsdp_enabled", lambda: False)
    monkeypatch.setattr(patcher_mod, "config_uses_qwen3_omni_moe_blocks", lambda config: False)
    monkeypatch.setattr(patcher_mod, "is_transformers_version_greater_than", lambda version: True)


def _enable_audio_output(freeze_talker: bool) -> bool:
    config = SimpleNamespace(model_type="qwen2_5_omni", enable_audio_output=None, architectures=None)
    model_args = SimpleNamespace(compute_dtype="unused", infer_dtype="auto", low_cpu_mem_usage=False, device_map=None)
    patcher_mod.patch_config(
        config,
        tokenizer=None,
        model_args=model_args,
        init_kwargs={"low_cpu_mem_usage": False},
        is_trainable=True,
        finetuning_args=SimpleNamespace(freeze_talker=freeze_talker, lora_language_model_experts=False),
    )
    return config.enable_audio_output


@pytest.mark.runs_on(["cpu", "mps"])
def test_patch_config_enables_audio_output_when_talker_trains(monkeypatch):
    _stub_patch_config_helpers(monkeypatch)
    assert _enable_audio_output(freeze_talker=False) is True
    assert _enable_audio_output(freeze_talker=True) is False
