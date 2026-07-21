# Copyright 2025 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

from llamafactory.model.model_utils.qwen3_omni_moe_lora import (
    Qwen3OmniMoeThinkerTextExpertsLora,
    adapter_requires_expert_lora,
    is_qwen3_omni_thinker_experts,
    load_peft_model_maybe_expert_lora,
    load_peft_model_with_expert_lora,
    register_qwen3_omni_moe_expert_lora,
    tag_experts_implementation,
)
from llamafactory.model.model_utils.visual import build_component_lora_targets


def _make_peft_model(host: nn.Module) -> PeftModel:
    """Wrap a tiny host without CausalLM task-type forward assumptions."""
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules="q_proj|experts",
        lora_dropout=0.0,
        task_type=None,
    )
    register_qwen3_omni_moe_expert_lora(config)
    return get_peft_model(host, config)


def _make_experts_config(num_experts: int = 4, hidden: int = 16, intermediate: int = 8):
    return SimpleNamespace(
        num_experts=num_experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        hidden_act="silu",
        _experts_implementation="eager",
    )


def _make_experts(num_experts: int = 4, hidden: int = 16, intermediate: int = 8) -> Qwen3OmniMoeThinkerTextExperts:
    cfg = _make_experts_config(num_experts, hidden, intermediate)
    experts = Qwen3OmniMoeThinkerTextExperts(cfg)
    tag_experts_implementation(experts, "eager")
    nn.init.normal_(experts.gate_up_proj, std=0.02)
    nn.init.normal_(experts.down_proj, std=0.02)
    return experts


def _route(hidden: torch.Tensor, num_experts: int = 4, top_k: int = 2):
    tokens = hidden.shape[0]
    top_k_index = torch.stack(
        [
            torch.arange(tokens) % num_experts,
            (torch.arange(tokens) + 1) % num_experts,
        ],
        dim=-1,
    )
    top_k_weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=hidden.dtype)
    return top_k_index, top_k_weights


def _reference_expert_output(
    experts: Qwen3OmniMoeThinkerTextExperts,
    hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    lora_layer: Qwen3OmniMoeThinkerTextExpertsLora | None = None,
    adapter_name: str = "default",
) -> torch.Tensor:
    """Explicit effective-weight reference for sparse expert (+ optional LoRA) forward."""
    out = torch.zeros_like(hidden)
    num_experts = experts.num_experts
    expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = int(expert_idx[0])
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        x = hidden[token_idx]
        gate_up = experts.gate_up_proj[expert_idx]
        down = experts.down_proj[expert_idx]
        if lora_layer is not None and adapter_name in lora_layer.lora_A_gate:
            dg, dd = lora_layer.get_delta_weight_for_expert(adapter_name, expert_idx)
            gate_up = gate_up + dg.to(gate_up.dtype)
            down = down + dd.to(down.dtype)
        gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
        y = F.linear(experts.act_fn(gate) * up, down)
        y = y * top_k_weights[token_idx, top_k_pos, None]
        out.index_add_(0, token_idx, y.to(out.dtype))
    return out


class _Config(dict):
    """Dict config that also supports attribute access (PEFT + COMPOSITE_MODELS)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _TinyThinker(nn.Module):
    def __init__(self, n_layers: int = 1):
        super().__init__()
        self.config = _Config(model_type="speechlmm", tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(16, 16, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.experts = _make_experts()
            self.model.layers.append(layer)
        self.lm_head = nn.Linear(16, 32, bias=False)


class _TinyPeftHost(nn.Module):
    """Minimal host model that PEFT can wrap for save/load tests."""

    def __init__(self):
        super().__init__()
        self.config = _Config(model_type="speechlmm", tie_word_embeddings=False)
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.experts = _make_experts()

    def forward(self, x, top_k_index, top_k_weights):
        x = self.q_proj(x)
        return self.experts(x, top_k_index, top_k_weights)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


def test_is_qwen3_omni_thinker_experts():
    assert is_qwen3_omni_thinker_experts(_make_experts())
    assert not is_qwen3_omni_thinker_experts(nn.Linear(4, 4))


def test_zero_init_matches_base():
    torch.manual_seed(0)
    experts = _make_experts()
    hidden = torch.randn(6, 16)
    top_k_index, top_k_weights = _route(hidden)
    base_out = experts(hidden, top_k_index, top_k_weights)

    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    lora_out = layer(hidden, top_k_index, top_k_weights)
    assert torch.allclose(base_out, lora_out, atol=1e-5, rtol=1e-5)


def test_routed_output_matches_effective_weight_reference():
    torch.manual_seed(1)
    experts = _make_experts()
    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    with torch.no_grad():
        layer.lora_B_gate["default"].add_(0.05)
        layer.lora_B_up["default"].add_(0.04)
        layer.lora_B_down["default"].add_(0.03)

    hidden = torch.randn(5, 16)
    top_k_index, top_k_weights = _route(hidden)
    out = layer(hidden, top_k_index, top_k_weights)
    ref = _reference_expert_output(
        experts, hidden, top_k_index, top_k_weights, lora_layer=layer
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_only_selected_experts_get_gradients():
    torch.manual_seed(2)
    experts = _make_experts(num_experts=4)
    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    with torch.no_grad():
        layer.lora_B_gate["default"].add_(0.1)
        layer.lora_B_up["default"].add_(0.1)
        layer.lora_B_down["default"].add_(0.1)
    hidden = torch.randn(3, 16)
    top_k_index = torch.tensor([[0, 1], [0, 1], [0, 1]])
    top_k_weights = torch.full((3, 2), 0.5)
    out = layer(hidden, top_k_index, top_k_weights)
    out.sum().backward()

    for name in ("lora_A_gate", "lora_B_gate", "lora_A_up", "lora_B_up", "lora_A_down", "lora_B_down"):
        grad = getattr(layer, name)["default"].grad
        assert grad is not None, name
        selected = grad[:2].abs().sum().item()
        unused = grad[2:].abs().sum().item()
        assert selected > 0.0, name
        assert unused == 0.0, name


def test_zero_b_initial_step_has_zero_a_gradients():
    torch.manual_seed(21)
    experts = _make_experts(num_experts=4)
    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    hidden = torch.randn(3, 16)
    top_k_index = torch.tensor([[0, 1], [0, 1], [0, 1]])
    top_k_weights = torch.full((3, 2), 0.5)
    layer(hidden, top_k_index, top_k_weights).sum().backward()
    for name in ("lora_A_gate", "lora_A_up", "lora_A_down"):
        assert getattr(layer, name)["default"].grad.abs().sum().item() == 0.0, name
    for name in ("lora_B_gate", "lora_B_up", "lora_B_down"):
        assert getattr(layer, name)["default"].grad[:2].abs().sum().item() > 0.0, name


def test_parameter_count_matches_three_rank4_projections():
    experts = _make_experts(num_experts=4, hidden=16, intermediate=8)
    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    trainable = 0
    for name in layer.adapter_layer_names:
        trainable += getattr(layer, name)["default"].numel()
    expected = (
        2 * (4 * 4 * 16 + 4 * 8 * 4)  # gate + up
        + (4 * 4 * 8 + 4 * 16 * 4)  # down
    )
    assert trainable == expected


def test_multi_layer_target_count():
    model = _TinyThinker(n_layers=2)
    args = SimpleNamespace(
        lora_audio_encoder=False,
        lora_audio_adapters=False,
        lora_language_model=True,
        lora_language_model_rank=8,
        lora_language_model_alpha=16,
        lora_language_model_experts=True,
        lora_language_model_experts_rank=4,
        lora_language_model_experts_alpha=8,
        lora_lipread_encoder=False,
        lora_lipread_adapter=False,
        lora_rank=8,
        lora_alpha=16,
    )
    targets, _, _ = build_component_lora_targets(model, args)
    expert_targets = [t for t in targets if t.endswith("mlp.experts")]
    assert expert_targets == ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]


def test_fp32_and_bf16_base_fp32_adapter_forward_backward():
    torch.manual_seed(7)
    for base_dtype in (torch.float32, torch.bfloat16):
        experts = _make_experts()
        experts.gate_up_proj.data = experts.gate_up_proj.data.to(base_dtype)
        experts.down_proj.data = experts.down_proj.data.to(base_dtype)
        layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
        assert layer.lora_A_gate["default"].dtype == torch.float32
        with torch.no_grad():
            layer.lora_B_gate["default"].add_(0.05)
            layer.lora_B_up["default"].add_(0.04)
            layer.lora_B_down["default"].add_(0.03)
        hidden = torch.randn(4, 16, dtype=base_dtype)
        top_k_index, top_k_weights = _route(hidden)
        top_k_weights = top_k_weights.to(base_dtype)
        out = layer(hidden, top_k_index, top_k_weights)
        assert out.dtype == base_dtype
        out.float().sum().backward()
        assert layer.lora_B_gate["default"].grad is not None


def test_dropout_shares_gate_up_mask():
    torch.manual_seed(8)
    experts = _make_experts()
    layer = Qwen3OmniMoeThinkerTextExpertsLora(
        experts, "default", r=4, lora_alpha=8, lora_dropout=0.5
    )
    with torch.no_grad():
        layer.lora_B_gate["default"].fill_(0.2)
        layer.lora_B_up["default"].fill_(0.2)
        layer.lora_A_gate["default"].fill_(0.1)
        layer.lora_A_up["default"].fill_(0.1)

    hidden = torch.ones(2, 16)
    top_k_index = torch.tensor([[0, 1], [0, 1]])
    top_k_weights = torch.full((2, 2), 0.5)
    layer.train()
    masks: list[torch.Tensor] = []

    class _TrackingDropout(nn.Module):
        def __init__(self, p: float):
            super().__init__()
            self.p = p

        def forward(self, x):
            keep = (torch.rand_like(x) > self.p).to(x.dtype) / (1.0 - self.p)
            masks.append(keep.detach().clone())
            return x * keep

    layer.lora_dropout["default"] = _TrackingDropout(0.5)
    torch.manual_seed(123)
    _ = layer(hidden, top_k_index, top_k_weights)
    # 2 hit experts * (1 shared gate/up + 1 down) = 4 dropout calls
    assert len(masks) == 4


def test_disable_adapters_matches_base():
    torch.manual_seed(3)
    experts = _make_experts()
    layer = Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)
    with torch.no_grad():
        layer.lora_B_gate["default"].add_(0.1)
    hidden = torch.randn(4, 16)
    top_k_index, top_k_weights = _route(hidden)
    base_out = experts(hidden, top_k_index, top_k_weights)
    layer.enable_adapters(False)
    assert torch.allclose(layer(hidden, top_k_index, top_k_weights), base_out, atol=1e-5)


def test_merge_unmerge_and_merge_unload_parity():
    torch.manual_seed(4)
    host = _TinyPeftHost()
    hidden = torch.randn(4, 16)
    top_k_index, top_k_weights = _route(hidden)

    model = _make_peft_model(host)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name and param.requires_grad:
                param.add_(0.05)

    adapted = model(hidden, top_k_index, top_k_weights).detach()
    # safe merge + unmerge roundtrip on the wrapped experts module
    experts_lora = next(m for m in model.modules() if isinstance(m, Qwen3OmniMoeThinkerTextExpertsLora))
    experts_lora.merge(safe_merge=True)
    merged_fwd = model(hidden, top_k_index, top_k_weights).detach()
    assert torch.allclose(adapted, merged_fwd, atol=1e-5, rtol=1e-5)
    experts_lora.unmerge()
    unmerged_fwd = model(hidden, top_k_index, top_k_weights).detach()
    assert torch.allclose(adapted, unmerged_fwd, atol=1e-5, rtol=1e-5)

    merged = model.merge_and_unload()
    merged_out = merged(hidden, top_k_index, top_k_weights).detach()
    assert torch.allclose(adapted, merged_out, atol=1e-5, rtol=1e-5)
    assert not any("lora_" in k for k in merged.state_dict())


def test_build_component_targets_respects_experts_flag():
    model = _TinyThinker()
    args = SimpleNamespace(
        lora_audio_encoder=False,
        lora_audio_adapters=False,
        lora_language_model=True,
        lora_language_model_rank=16,
        lora_language_model_alpha=32,
        lora_language_model_experts=False,
        lora_language_model_experts_rank=4,
        lora_language_model_experts_alpha=8,
        lora_lipread_encoder=False,
        lora_lipread_adapter=False,
        lora_rank=8,
        lora_alpha=16,
    )
    targets, _, _ = build_component_lora_targets(model, args)
    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.0.mlp.experts" not in targets

    args.lora_language_model_experts = True
    targets, rank_pattern, alpha_pattern = build_component_lora_targets(model, args)
    assert "model.layers.0.mlp.experts" in targets
    assert rank_pattern["model.layers.0.mlp.experts"] == 4
    assert alpha_pattern["model.layers.0.mlp.experts"] == 8


def test_zero_expert_targets_raise():
    model = nn.Module()
    model.config = _Config(model_type="speechlmm", tie_word_embeddings=False)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].self_attn = nn.Module()
    model.model.layers[0].self_attn.q_proj = nn.Linear(16, 16, bias=False)
    args = SimpleNamespace(
        lora_audio_encoder=False,
        lora_audio_adapters=False,
        lora_language_model=True,
        lora_language_model_rank=8,
        lora_language_model_alpha=16,
        lora_language_model_experts=True,
        lora_language_model_experts_rank=4,
        lora_language_model_experts_alpha=8,
        lora_lipread_encoder=False,
        lora_lipread_adapter=False,
        lora_rank=8,
        lora_alpha=16,
    )
    with pytest.raises(ValueError, match="no Qwen3OmniMoeThinkerTextExperts"):
        build_component_lora_targets(model, args)


def test_rejects_non_eager_experts_implementation():
    experts = _make_experts()
    tag_experts_implementation(experts, "grouped_mm")
    with pytest.raises(ValueError, match="eager expert execution"):
        Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8)


def test_rejects_dora_and_pissa_init():
    experts = _make_experts()
    with pytest.raises(ValueError, match="use_dora"):
        Qwen3OmniMoeThinkerTextExpertsLora(experts, "default", r=4, lora_alpha=8, use_dora=True)
    with pytest.raises(ValueError, match="init_lora_weights"):
        Qwen3OmniMoeThinkerTextExpertsLora(
            experts, "default", r=4, lora_alpha=8, init_lora_weights="pissa"
        )


def test_register_custom_module_roundtrip_nonzero_factors():
    torch.manual_seed(5)
    host = _TinyPeftHost()
    base_state = {k: v.detach().clone() for k, v in host.state_dict().items()}
    model = _make_peft_model(host)
    assert any(isinstance(m, Qwen3OmniMoeThinkerTextExpertsLora) for m in model.modules())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(k in name for k in ("lora_A_gate", "lora_B_gate", "lora_A_up", "lora_B_up", "lora_A_down", "lora_B_down")):
                param.add_(0.01)

    hidden = torch.randn(3, 16)
    top_k_index, top_k_weights = _route(hidden)
    out1 = model(hidden, top_k_index, top_k_weights)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        cfg = json.loads(Path(tmp, "adapter_config.json").read_text())
        assert adapter_requires_expert_lora(tmp)
        # rank/alpha patterns persist for expert targets when set; target_modules must mention experts
        assert "experts" in str(cfg.get("target_modules", ""))

        host2 = _TinyPeftHost()
        host2.load_state_dict(base_state)
        reloaded = load_peft_model_with_expert_lora(host2, tmp)
        out2 = reloaded(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out1, out2, atol=1e-5, rtol=1e-5)

        # Conditional loader also registers expert mapping when targets include experts.
        host3 = _TinyPeftHost()
        host3.load_state_dict(base_state)
        reloaded2 = load_peft_model_maybe_expert_lora(host3, tmp)
        out3 = reloaded2(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out1, out3, atol=1e-5, rtol=1e-5)


def test_adapter_requires_expert_lora_detection():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "target_modules": r"model\.layers\.0\.self_attn\.q_proj|model\.layers\.0\.mlp\.experts",
                    "rank_pattern": {"model.layers.0.mlp.experts": 4},
                    "alpha_pattern": {"model.layers.0.mlp.experts": 8},
                }
            )
        )
        assert adapter_requires_expert_lora(path)

        (path / "adapter_config.json").write_text(
            json.dumps({"target_modules": r"model\.layers\.0\.self_attn\.q_proj"})
        )
        assert not adapter_requires_expert_lora(path)
