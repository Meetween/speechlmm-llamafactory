# Copyright 2025 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from llamafactory.hparams import FinetuningArguments
from llamafactory.model.adapter import _peft_exact_target_modules
from llamafactory.model.model_utils.misc import find_all_linear_modules
from llamafactory.model.model_utils.visual import (
    build_component_lora_targets,
    patch_target_modules,
)


class _Config(dict):
    """Dict config that also supports attribute access (PEFT + COMPOSITE_MODELS)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _TinyDenseSpeechLMM(nn.Module):
    """Dense Thinker + Talker stand-in for SpeechLMM-v2 L LoRA targeting."""

    def __init__(self):
        super().__init__()
        self.config = _Config(model_type="speechlmm", tie_word_embeddings=False)
        self.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(8, 16, bias=False)
        layer.mlp.up_proj = nn.Linear(8, 16, bias=False)
        layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
        layer.mlp.seq = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 8, bias=False),
        )
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.talker = nn.Module()
        self.talker.model = nn.Module()
        talker_layer = nn.Module()
        talker_layer.self_attn = nn.Module()
        talker_layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.talker.model.layers = nn.ModuleList([talker_layer])


def _component_args(**overrides) -> SimpleNamespace:
    args = dict(
        lora_audio_encoder=False,
        lora_audio_adapters=False,
        lora_language_model=True,
        lora_language_model_scope="attention",
        lora_language_model_rank=8,
        lora_language_model_alpha=16,
        lora_language_model_experts=False,
        lora_language_model_experts_rank=4,
        lora_language_model_experts_alpha=8,
        lora_lipread_encoder=False,
        lora_lipread_adapter=False,
        lora_talker=False,
        lora_rank=8,
        lora_alpha=16,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _speechlmm_l7b_dir() -> Path | None:
    env = os.environ.get("SPEECHLMM_L7B_SOURCE")
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    models_dir = os.environ.get("MODELS_DIR")
    if not models_dir:
        return None
    for rel in ("SpeechLMM-v2.0-L-7B", "SpeechLMM-v2/SpeechLMM-v2.0-L-7B"):
        path = Path(models_dir) / rel
        if path.is_dir():
            return path
    return None


def test_patch_target_all_does_not_select_decoder_layer():
    model = _TinyDenseSpeechLMM()
    args = FinetuningArguments(finetuning_type="lora")
    names = find_all_linear_modules(model, freeze_vision_tower=True, freeze_audio_tower=True)
    patched = patch_target_modules(model, args, names)
    module_map = dict(model.named_modules())
    assert "model.layers.0" not in patched
    assert "model.layers.0.self_attn.q_proj" in patched
    assert "model.layers.0.mlp.gate_proj" in patched
    assert "talker.model.layers.0.self_attn.q_proj" not in patched
    for name in patched:
        assert isinstance(module_map[name], nn.Linear)


def test_numeric_target_does_not_select_decoder_layer():
    model = _TinyDenseSpeechLMM()
    args = FinetuningArguments(finetuning_type="lora")
    patched = patch_target_modules(model, args, ["0"])
    assert "model.layers.0" not in patched
    assert all(name.endswith(".0") for name in patched)
    module_map = dict(model.named_modules())
    for name in patched:
        assert isinstance(module_map[name], nn.Linear)


def test_lora_target_all_peft_wrap_succeeds():
    model = _TinyDenseSpeechLMM()
    args = FinetuningArguments(finetuning_type="lora")
    names = find_all_linear_modules(model, freeze_vision_tower=True, freeze_audio_tower=True)
    patched = patch_target_modules(model, args, names)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=_peft_exact_target_modules(patched),
        lora_dropout=0.0,
        task_type=None,
    )
    peft_model = get_peft_model(model, config)
    lora_keys = [name for name, _ in peft_model.named_parameters() if "lora_" in name]
    assert any(name.endswith("q_proj.lora_A.default.weight") for name in lora_keys)
    assert not any(re.search(r"layers\.0\.lora_", name) for name in lora_keys)


def test_attention_scope_excludes_dense_ffn():
    model = _TinyDenseSpeechLMM()
    targets, _, _ = build_component_lora_targets(model, _component_args())
    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.0.self_attn.k_proj" in targets
    assert "model.layers.0.self_attn.v_proj" in targets
    assert "model.layers.0.self_attn.o_proj" in targets
    assert "lm_head" in targets
    assert "model.layers.0.mlp.gate_proj" not in targets
    assert "model.layers.0.mlp.up_proj" not in targets
    assert "model.layers.0.mlp.down_proj" not in targets
    assert "talker.model.layers.0.self_attn.q_proj" not in targets
    assert "model.layers.0.mlp.seq.0" not in targets


def test_legacy_language_model_flag_is_attention_only():
    model = _TinyDenseSpeechLMM()
    targets, _, _ = build_component_lora_targets(model, _component_args(lora_language_model_scope=None))
    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.0.mlp.gate_proj" not in targets


@pytest.mark.skipif(_speechlmm_l7b_dir() is None, reason="set SPEECHLMM_L7B_SOURCE or MODELS_DIR")
def test_real_speechlmm_l7b_lora_target_all_and_attention_scope():
    from transformers import AutoConfig, AutoModel

    src = _speechlmm_l7b_dir()
    config = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModel.from_config(config, trust_remote_code=True)

    names = find_all_linear_modules(model, freeze_vision_tower=True, freeze_audio_tower=True)
    args = FinetuningArguments(finetuning_type="lora")
    patched = patch_target_modules(model, args, names)
    module_map = dict(model.named_modules())
    assert "model.layers.0" not in patched
    for name in patched:
        assert isinstance(module_map[name], nn.Linear)

    attn, _, _ = build_component_lora_targets(model, _component_args())
    attn_leaves = {name.rsplit(".", 1)[-1] for name in attn}
    assert attn_leaves <= {"q_proj", "k_proj", "v_proj", "o_proj", "lm_head"}
    assert "gate_proj" not in attn_leaves

    ffn, _, _ = build_component_lora_targets(model, _component_args(lora_language_model_scope="attention_and_ffn"))
    ffn_leaves = {name.rsplit(".", 1)[-1] for name in ffn}
    assert {"q_proj", "gate_proj", "up_proj", "down_proj", "lm_head"} <= ffn_leaves

    peft_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=_peft_exact_target_modules(patched),
        lora_dropout=0.0,
        task_type=None,
    )
    get_peft_model(model, peft_cfg)


def test_attention_and_ffn_includes_dense_mlp():
    model = _TinyDenseSpeechLMM()
    targets, _, _ = build_component_lora_targets(model, _component_args(lora_language_model_scope="attention_and_ffn"))
    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.0.mlp.gate_proj" in targets
    assert "model.layers.0.mlp.up_proj" in targets
    assert "model.layers.0.mlp.down_proj" in targets
    assert "lm_head" in targets
    assert "model.layers.0.mlp.seq.0" not in targets
    assert "talker.model.layers.0.self_attn.q_proj" not in targets
