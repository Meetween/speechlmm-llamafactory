# Copyright 2025 the LlamaFactory / SpeechLMM team.
#
# Custom PEFT LoRA layer for Qwen3-Omni Thinker fused MoE experts.
# Applies independent per-expert LoRA to gate/up/down while keeping the
# native fused Parameter layout and sparse top-k expert dispatch.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType
from peft.tuners.lora.layer import LoraLayer
from peft.utils.integrations import gather_params_ctx

from ...extras import logging


logger = logging.get_logger(__name__)

_SUPPORTED_EXPERTS_IMPL = (None, "eager")
_SUPPORTED_INIT = {True, False, "gaussian"}
_EXPERT_TARGET_MARKERS = ("mlp.experts",)


def _get_thinker_text_experts_cls():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextExperts,
    )

    return Qwen3OmniMoeThinkerTextExperts


def is_qwen3_omni_thinker_experts(module: nn.Module) -> bool:
    try:
        experts_cls = _get_thinker_text_experts_cls()
    except Exception:
        return False
    return isinstance(module, experts_cls)


def resolve_experts_implementation(module: nn.Module) -> str | None:
    """Return the effective experts backend for a Thinker experts module."""
    impl = getattr(module, "_experts_implementation", None)
    if impl is not None:
        return impl
    config = getattr(module, "config", None)
    return getattr(config, "_experts_implementation", None)


def tag_experts_implementation(module: nn.Module, experts_impl: str | None) -> None:
    module._experts_implementation = experts_impl


def adapter_requires_expert_lora(adapter_path: str | Path) -> bool:
    """Detect expert LoRA from standard PEFT target / rank-pattern / factor keys."""
    adapter_dir = Path(adapter_path)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return False

    with open(cfg_path) as fh:
        cfg = json.load(fh)

    def _has_expert_marker(value: str) -> bool:
        cleaned = value.replace("\\", "")
        if any(marker in cleaned for marker in _EXPERT_TARGET_MARKERS):
            return True
        # PEFT may persist short module names such as "experts" for regex targets.
        parts = cleaned.split("|") if "|" in cleaned else [cleaned]
        return any(part == "experts" or part.endswith(".experts") for part in parts)

    tm = cfg.get("target_modules", "")
    if isinstance(tm, str) and _has_expert_marker(tm):
        return True
    if isinstance(tm, (list, set, tuple)) and any(_has_expert_marker(str(t)) for t in tm):
        return True

    for key in ("rank_pattern", "alpha_pattern"):
        pattern = cfg.get(key) or {}
        if isinstance(pattern, dict) and any(_has_expert_marker(str(k)) for k in pattern):
            return True

    # Custom expert factors are never produced by ordinary Linear LoRA.
    weight_path = adapter_dir / "adapter_model.safetensors"
    if weight_path.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if any(
                        token in key
                        for token in (
                            "lora_A_gate",
                            "lora_B_gate",
                            "lora_A_up",
                            "lora_B_up",
                            "lora_A_down",
                            "lora_B_down",
                        )
                    ):
                        return True
        except Exception:
            pass

    return False


def register_qwen3_omni_moe_expert_lora(peft_config: LoraConfig) -> LoraConfig:
    """Register the custom Experts LoRA layer on a LoraConfig instance."""
    experts_cls = _get_thinker_text_experts_cls()
    peft_config._register_custom_module({experts_cls: Qwen3OmniMoeThinkerTextExpertsLora})
    return peft_config


def force_eager_experts_on_model(model: nn.Module) -> None:
    """Force Transformers expert backends to eager before wrapping/loading expert LoRA."""
    from transformers import PreTrainedModel as HFPreTrainedModel

    if hasattr(model, "set_experts_implementation"):
        try:
            model.set_experts_implementation("eager")
        except Exception as exc:
            logger.warning_rank0(f"Could not set eager experts on root model: {exc}")

    for submodule in model.modules():
        if (
            submodule is not model
            and isinstance(submodule, HFPreTrainedModel)
            and hasattr(submodule, "set_experts_implementation")
        ):
            try:
                submodule.set_experts_implementation("eager")
            except Exception as exc:
                logger.warning_rank0(
                    f"Could not set eager experts on {type(submodule).__name__}: {exc}"
                )
        if is_qwen3_omni_thinker_experts(submodule):
            tag_experts_implementation(submodule, "eager")
            expert_cfg = getattr(submodule, "config", None)
            if expert_cfg is not None:
                expert_cfg._experts_implementation = "eager"


def load_peft_model_with_expert_lora(
    model: nn.Module,
    adapter_path: str,
    *,
    is_trainable: bool = False,
    **kwargs,
) -> PeftModel:
    """Load a PEFT adapter, re-registering the Experts custom module mapping."""
    force_eager_experts_on_model(model)
    peft_config = LoraConfig.from_pretrained(adapter_path)
    register_qwen3_omni_moe_expert_lora(peft_config)
    return PeftModel.from_pretrained(
        model,
        adapter_path,
        config=peft_config,
        is_trainable=is_trainable,
        **kwargs,
    )


def load_peft_model_maybe_expert_lora(
    model: nn.Module,
    adapter_path: str,
    *,
    is_trainable: bool = False,
    **kwargs,
) -> PeftModel:
    """Load PEFT adapter; register Experts mapping only when the adapter needs it."""
    if adapter_requires_expert_lora(adapter_path):
        return load_peft_model_with_expert_lora(
            model, adapter_path, is_trainable=is_trainable, **kwargs
        )
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable, **kwargs)


def create_lora_config_with_expert_lora(**peft_kwargs) -> LoraConfig:
    """Build a LoraConfig and register the Experts custom module mapping."""
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs)
    return register_qwen3_omni_moe_expert_lora(peft_config)


def _lora_delta(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute scale * B(A(x)) with LoRA factor dtype, returning activation dtype."""
    if x.numel() == 0:
        return x.new_zeros(x.shape[0], b.shape[0])
    x_lora = x.to(dtype=a.dtype)
    delta = scaling * F.linear(F.linear(x_lora, a), b)
    return delta.to(dtype=x.dtype)


class Qwen3OmniMoeThinkerTextExpertsLora(nn.Module, LoraLayer):
    """Sparse per-expert LoRA wrapper for fused Qwen3-Omni Thinker experts.

    Base weights stay as fused ``gate_up_proj`` / ``down_proj`` Parameters.
    Trainable LoRA factors are stored per expert for gate, up, and down.
    Forward evaluates LoRA only for experts selected by top-k routing.
    """

    adapter_layer_names = (
        "lora_A_gate",
        "lora_B_gate",
        "lora_A_up",
        "lora_B_up",
        "lora_A_down",
        "lora_B_down",
    )

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if not is_qwen3_omni_thinker_experts(base_layer):
            raise TypeError(
                "Qwen3OmniMoeThinkerTextExpertsLora expects Qwen3OmniMoeThinkerTextExperts, "
                f"got {type(base_layer).__name__}."
            )
        if use_dora:
            raise ValueError("Qwen3OmniMoeThinkerTextExpertsLora does not support use_dora=True.")
        if lora_bias:
            raise ValueError("Qwen3OmniMoeThinkerTextExpertsLora does not support lora_bias=True.")

        experts_impl = resolve_experts_implementation(base_layer)
        if experts_impl not in _SUPPORTED_EXPERTS_IMPL:
            raise ValueError(
                "lora_language_model_experts requires eager expert execution "
                f"(config._experts_implementation in {_SUPPORTED_EXPERTS_IMPL!r}), "
                f"got {experts_impl!r}."
            )

        unsupported = {
            "use_qalora",
            "use_alora",
            "alora_invocation_tokens",
            "arrow_config",
            "ensure_weight_tying",
        }
        for key in unsupported:
            if kwargs.get(key):
                raise ValueError(f"Qwen3OmniMoeThinkerTextExpertsLora does not support {key}={kwargs[key]!r}.")

        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self.num_experts = base_layer.num_experts
        self.hidden_dim = base_layer.hidden_dim
        self.intermediate_dim = base_layer.intermediate_dim
        self.act_fn = base_layer.act_fn

        self.lora_A_gate = nn.ParameterDict()
        self.lora_B_gate = nn.ParameterDict()
        self.lora_A_up = nn.ParameterDict()
        self.lora_B_up = nn.ParameterDict()
        self.lora_A_down = nn.ParameterDict()
        self.lora_B_down = nn.ParameterDict()

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=False,
            lora_bias=False,
        )

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if use_dora:
            raise ValueError("Qwen3OmniMoeThinkerTextExpertsLora does not support use_dora=True.")
        if lora_bias:
            raise ValueError("Qwen3OmniMoeThinkerTextExpertsLora does not support lora_bias=True.")
        if self.resolve_lora_variant(use_dora=use_dora) is not None:
            raise ValueError("Qwen3OmniMoeThinkerTextExpertsLora does not support LoRA variants like DoRA.")
        if init_lora_weights not in _SUPPORTED_INIT:
            raise ValueError(
                "Qwen3OmniMoeThinkerTextExpertsLora only supports init_lora_weights "
                "in {True, False, 'gaussian'}."
            )

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout.update(nn.ModuleDict({adapter_name: nn.Dropout(p=lora_dropout)}))
        else:
            self.lora_dropout.update(nn.ModuleDict({adapter_name: nn.Identity()}))

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        self.use_rslora[adapter_name] = use_rslora
        self.lora_bias[adapter_name] = False

        device = self.base_layer.gate_up_proj.device
        # Keep trainable factors in fp32 even when the fused base is bf16/fp16.
        dtype = torch.float32

        self.lora_A_gate[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, r, self.hidden_dim, device=device, dtype=dtype)
        )
        self.lora_B_gate[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, r, device=device, dtype=dtype)
        )
        self.lora_A_up[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, r, self.hidden_dim, device=device, dtype=dtype)
        )
        self.lora_B_up[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, r, device=device, dtype=dtype)
        )
        self.lora_A_down[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, r, self.intermediate_dim, device=device, dtype=dtype)
        )
        self.lora_B_down[adapter_name] = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, r, device=device, dtype=dtype)
        )

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self.set_adapter(self.active_adapters, inference_mode=kwargs.get("inference_mode", False))

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str = True) -> None:
        if init_lora_weights is False:
            return
        if isinstance(init_lora_weights, str) and init_lora_weights != "gaussian":
            raise ValueError(
                "Qwen3OmniMoeThinkerTextExpertsLora only supports init_lora_weights "
                "in {True, False, 'gaussian'}."
            )

        for a_name in ("lora_A_gate", "lora_A_up", "lora_A_down"):
            a_param = getattr(self, a_name)[adapter_name]
            if init_lora_weights == "gaussian":
                nn.init.normal_(a_param, std=1.0 / self.r[adapter_name])
            else:
                nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))

        for b_name in ("lora_B_gate", "lora_B_up", "lora_B_down"):
            nn.init.zeros_(getattr(self, b_name)[adapter_name])

    def _expert_forward(
        self,
        current_state: torch.Tensor,
        expert_idx: torch.Tensor,
        active_adapters: list[str],
    ) -> torch.Tensor:
        base = self.get_base_layer()
        # Preserve the native single fused gate_up GEMM, then split.
        gate_up = F.linear(current_state, base.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)

        if self.disable_adapters or self.merged or not active_adapters:
            return F.linear(self.act_fn(gate) * up, base.down_proj[expert_idx])

        for adapter_name in active_adapters:
            if adapter_name not in self.lora_A_gate:
                continue
            scaling = self.scaling[adapter_name]
            dropout = self.lora_dropout[adapter_name]
            # One dropped input shared by gate and up (same mask when dropout > 0).
            lora_in = dropout(current_state)
            gate = gate + _lora_delta(
                lora_in,
                self.lora_A_gate[adapter_name][expert_idx],
                self.lora_B_gate[adapter_name][expert_idx],
                scaling,
            )
            up = up + _lora_delta(
                lora_in,
                self.lora_A_up[adapter_name][expert_idx],
                self.lora_B_up[adapter_name][expert_idx],
                scaling,
            )

        hidden = self.act_fn(gate) * up
        output = F.linear(hidden, base.down_proj[expert_idx])
        for adapter_name in active_adapters:
            if adapter_name not in self.lora_A_down:
                continue
            scaling = self.scaling[adapter_name]
            dropout = self.lora_dropout[adapter_name]
            output = output + _lora_delta(
                dropout(hidden),
                self.lora_A_down[adapter_name][expert_idx],
                self.lora_B_down[adapter_name][expert_idx],
                scaling,
            )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        active_adapters = list(self.active_adapters)
        for expert_idx in expert_hit:
            # Keep a 0-dim tensor index (same as upstream); avoid int() host sync.
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._expert_forward(current_state, expert_idx, active_adapters)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def get_delta_weight_for_expert(self, adapter_name: str, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta_gate_up [2I,H], delta_down [H,I]) for one expert."""
        scaling = self.scaling[adapter_name]
        a_gate = self.lora_A_gate[adapter_name][expert_idx]
        b_gate = self.lora_B_gate[adapter_name][expert_idx]
        a_up = self.lora_A_up[adapter_name][expert_idx]
        b_up = self.lora_B_up[adapter_name][expert_idx]
        a_down = self.lora_A_down[adapter_name][expert_idx]
        b_down = self.lora_B_down[adapter_name][expert_idx]

        delta_gate = (b_gate @ a_gate) * scaling
        delta_up = (b_up @ a_up) * scaling
        delta_gate_up = torch.cat([delta_gate, delta_up], dim=0)
        delta_down = (b_down @ a_down) * scaling
        return delta_gate_up, delta_down

    def _gather_adapter_factors(self, adapter_name: str):
        contexts = []
        for name in self.adapter_layer_names:
            param = getattr(self, name)[adapter_name]
            contexts.append(gather_params_ctx(param))
        return contexts

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        from contextlib import ExitStack

        from peft.tuners.tuners_utils import check_adapters_to_merge

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base = self.get_base_layer()
        with ExitStack() as stack:
            stack.enter_context(gather_params_ctx(base.gate_up_proj))
            stack.enter_context(gather_params_ctx(base.down_proj))
            for adapter_name in adapter_names:
                if adapter_name not in self.lora_A_gate:
                    continue
                for ctx in self._gather_adapter_factors(adapter_name):
                    stack.enter_context(ctx)
                for expert_idx in range(self.num_experts):
                    delta_gate_up, delta_down = self.get_delta_weight_for_expert(adapter_name, expert_idx)
                    delta_gate_up = delta_gate_up.to(base.gate_up_proj.dtype)
                    delta_down = delta_down.to(base.down_proj.dtype)
                    if safe_merge:
                        gate_up = base.gate_up_proj.data[expert_idx].clone() + delta_gate_up
                        down = base.down_proj.data[expert_idx].clone() + delta_down
                        if not torch.isfinite(gate_up).all() or not torch.isfinite(down).all():
                            raise ValueError(
                                f"NaNs detected while merging expert LoRA adapter {adapter_name}."
                            )
                        base.gate_up_proj.data[expert_idx] = gate_up
                        base.down_proj.data[expert_idx] = down
                    else:
                        base.gate_up_proj.data[expert_idx] += delta_gate_up
                        base.down_proj.data[expert_idx] += delta_down
                self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        from contextlib import ExitStack

        if not self.merged:
            return
        base = self.get_base_layer()
        with ExitStack() as stack:
            stack.enter_context(gather_params_ctx(base.gate_up_proj))
            stack.enter_context(gather_params_ctx(base.down_proj))
            while self.merged_adapters:
                adapter_name = self.merged_adapters.pop()
                if adapter_name not in self.lora_A_gate:
                    continue
                for ctx in self._gather_adapter_factors(adapter_name):
                    stack.enter_context(ctx)
                for expert_idx in range(self.num_experts):
                    delta_gate_up, delta_down = self.get_delta_weight_for_expert(adapter_name, expert_idx)
                    base.gate_up_proj.data[expert_idx] -= delta_gate_up.to(base.gate_up_proj.dtype)
                    base.down_proj.data[expert_idx] -= delta_down.to(base.down_proj.dtype)

    def unload_and_optionally_merge_module(
        self,
        merge: bool,
        safe_merge: bool,
        adapter_names: Optional[list[str]],
    ) -> nn.Module:
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        return self.get_base_layer()
