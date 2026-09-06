# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import transformers
import transformers.models
from transformers.activations import ACT2FN

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
transformers_logger = transformers.utils.logging.get_logger(__name__)


@dataclass
class CompositeModel:
    model_type: str
    projector_key: str
    vision_model_keys: list[str]
    language_model_keys: list[str]
    lora_conflict_keys: list[str]
    audio_model_keys: list[str]
    audio_adapter_prefixes: list[str]
    lipread_model_keys: list[str]
    lipread_adapter_keys: list[str]
    talker_keys: list[str]
    code2wav_keys: list[str]
    # TODO: add talker_adapter_prefixes (for excluding code_predictor from talker LoRA)
    # TODO: add vision_adapter_prefixes (if vision tower gets encoder/adapter split)

    def get_projector(self, module: "torch.nn.Module") -> "torch.nn.Module":
        for key in self.projector_key.split("."):
            module = getattr(module, key)

        return module


COMPOSITE_MODELS: dict[str, "CompositeModel"] = {}


def _register_composite_model(
    model_type: str,
    projector_key: Optional[str] = None,
    vision_model_keys: Optional[list[str]] = None,
    language_model_keys: Optional[list[str]] = None,
    lora_conflict_keys: Optional[list[str]] = None,
    audio_model_keys: Optional[list[str]] = None,
    audio_adapter_prefixes: Optional[list[str]] = None,
    lipread_model_keys: Optional[list[str]] = None,
    lipread_adapter_keys: Optional[list[str]] = None,
    talker_keys: Optional[list[str]] = None,
    code2wav_keys: Optional[list[str]] = None,
):
    r"""Register a new composite model.

    Args:
        model_type: model type
        projector_key: multi_modal_projector
        vision_model_keys: vision_tower
        language_model_keys: language_model
        lora_conflict_keys: None
        audio_model_keys: audio encoder (separate from vision for independent freeze control)
        audio_adapter_prefixes: audio adapter sub-modules (e.g. proj1, proj2, ln_post)
        lipread_model_keys: lipread encoder sub-model
        lipread_adapter_keys: lipread adapter sub-model
        talker_keys: speech generation sub-model (e.g. SpeechLMM's Talker)
        code2wav_keys: waveform synthesis sub-model (e.g. SpeechLMM's Code2Wav)

    """
    COMPOSITE_MODELS[model_type] = CompositeModel(
        model_type=model_type,
        projector_key=projector_key or "multi_modal_projector",
        vision_model_keys=vision_model_keys or ["vision_tower"],
        language_model_keys=language_model_keys or ["language_model", "lm_head"],
        lora_conflict_keys=lora_conflict_keys or [],
        audio_model_keys=audio_model_keys or [],
        audio_adapter_prefixes=audio_adapter_prefixes or [],
        lipread_model_keys=lipread_model_keys or [],
        lipread_adapter_keys=lipread_adapter_keys or [],
        talker_keys=talker_keys or [],
        code2wav_keys=code2wav_keys or [],
    )


class LlavaMultiModalProjectorForYiVL(torch.nn.Module):
    def __init__(self, config: "LlavaConfig") -> None:
        super().__init__()

        self.config = config
        if config is None:
            return

        self.linear_1 = torch.nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: "torch.Tensor") -> "torch.Tensor":
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        if hidden_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.linear_1.weight.dtype

            transformers_logger.warning_once("The hidden states seems to be silently casted in float32.")
            hidden_states = hidden_states.to(target_dtype)

        return hidden_states


class LlavaMultiModalProjectorForYiVLForVLLM(LlavaMultiModalProjectorForYiVL):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_hidden_act: str) -> None:
        super().__init__(config=None)

        self.linear_1 = torch.nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Cast projector output to half precision for fine-tuning quantized VLMs."""

    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)
        if model_type in COMPOSITE_MODELS:
            mm_projector = COMPOSITE_MODELS[model_type].get_projector(model)
        else:
            return

        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def configure_visual_model(config: "PretrainedConfig") -> None:
    r"""Patch VLMs before loading them."""
    if getattr(config, "text_config", None) and not getattr(config, "hidden_size", None):
        # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))

    if getattr(config, "is_yi_vl_derived_model", None):
        logger.info_rank0("Detected Yi-VL model, applying projector patch.")
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> set[str]:
    r"""Freeze vision tower, language model, talker, and code2wav for VLM/SpeechLMM full/freeze tuning."""
    model_type = getattr(config, "model_type", None)
    forbidden_modules = set()
    if model_type in COMPOSITE_MODELS:
        composite = COMPOSITE_MODELS[model_type]

        if finetuning_args.freeze_vision_tower:
            logger.info_rank0(f"Set vision model not trainable: {composite.vision_model_keys}.")
            forbidden_modules.update(composite.vision_model_keys)

        if composite.audio_model_keys:
            if composite.audio_adapter_prefixes:
                freeze_enc = finetuning_args.freeze_audio_encoder
                freeze_adp = finetuning_args.freeze_audio_adapters
                if freeze_enc:
                    # audio_model_keys (e.g. "audio_tower") covers all sub-modules
                    # including adapters, so no need to also add adapter prefixes.
                    logger.info_rank0(f"Set audio encoder not trainable: {composite.audio_model_keys}.")
                    forbidden_modules.update(composite.audio_model_keys)
                if freeze_adp and not freeze_enc:
                    logger.info_rank0(f"Set audio adapters not trainable: {composite.audio_adapter_prefixes}.")
                    forbidden_modules.update(composite.audio_adapter_prefixes)
            elif finetuning_args.freeze_audio_tower:
                logger.info_rank0(f"Set audio model not trainable: {composite.audio_model_keys}.")
                forbidden_modules.update(composite.audio_model_keys)

        if finetuning_args.freeze_multi_modal_projector:
            logger.info_rank0(f"Set multi model projector not trainable: {composite.projector_key}.")
            forbidden_modules.add(composite.projector_key)

        if finetuning_args.freeze_language_model:
            logger.info_rank0(f"Set language model not trainable: {composite.language_model_keys}.")
            forbidden_modules.update(composite.language_model_keys)

        if finetuning_args.freeze_talker and composite.talker_keys:
            logger.info_rank0(f"Set talker not trainable: {composite.talker_keys}.")
            forbidden_modules.update(composite.talker_keys)

        if finetuning_args.freeze_code2wav and composite.code2wav_keys:
            logger.info_rank0(f"Set code2wav not trainable: {composite.code2wav_keys}.")
            forbidden_modules.update(composite.code2wav_keys)

        if finetuning_args.freeze_code_predictor:
            logger.info_rank0("Set code_predictor not trainable: ['talker.code_predictor'].")
            forbidden_modules.add("talker.code_predictor")

        if finetuning_args.freeze_lipread_encoder and composite.lipread_model_keys:
            logger.info_rank0(f"Set lipread encoder not trainable: {composite.lipread_model_keys}.")
            forbidden_modules.update(composite.lipread_model_keys)

        if finetuning_args.freeze_lipread_adapter and composite.lipread_adapter_keys:
            logger.info_rank0(f"Set lipread adapter not trainable: {composite.lipread_adapter_keys}.")
            forbidden_modules.update(composite.lipread_adapter_keys)

    return forbidden_modules


def _matches_prefix(name: str, prefixes: list[str] | None) -> bool:
    """Check if *name* starts with any of the given prefixes."""
    if not prefixes:
        return False
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def build_component_lora_targets(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Select LoRA targets and per-module rank/alpha from component flags.

    Returns:
        target_modules: full-path names of ``nn.Linear`` modules to wrap with LoRA.
        rank_pattern: ``{name: rank}`` for modules whose rank differs from the global default.
        alpha_pattern: ``{name: alpha}`` for modules whose alpha differs from the global default.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        return [], {}, {}

    composite = COMPOSITE_MODELS[model_type]

    ComponentSpec = tuple[str, int, int, list[str], list[str]]
    components: list[ComponentSpec] = []

    if finetuning_args.lora_audio_encoder:
        components.append(
            (
                "audio_encoder",
                finetuning_args.lora_audio_encoder_rank,
                finetuning_args.lora_audio_encoder_alpha,
                composite.audio_model_keys,
                composite.audio_adapter_prefixes,
            )
        )

    if finetuning_args.lora_audio_adapters:
        components.append(
            (
                "audio_adapters",
                finetuning_args.lora_audio_adapters_rank,
                finetuning_args.lora_audio_adapters_alpha,
                composite.audio_adapter_prefixes,
                [],
            )
        )

    if finetuning_args.lora_language_model:
        components.append(
            (
                "language_model",
                finetuning_args.lora_language_model_rank,
                finetuning_args.lora_language_model_alpha,
                composite.language_model_keys,
                [],
            )
        )

    if finetuning_args.lora_lipread_encoder:
        components.append(
            (
                "lipread_encoder",
                finetuning_args.lora_lipread_encoder_rank,
                finetuning_args.lora_lipread_encoder_alpha,
                composite.lipread_model_keys,
                [],
            )
        )

    if finetuning_args.lora_lipread_adapter:
        components.append(
            (
                "lipread_adapter",
                finetuning_args.lora_lipread_adapter_rank,
                finetuning_args.lora_lipread_adapter_alpha,
                composite.lipread_adapter_keys,
                [],
            )
        )

    # TODO: add lora_talker block (composite.talker_keys, exclude code_predictor heads?)
    # TODO: add lora_vision_encoder block (composite.vision_model_keys)
    # TODO(low-priority): add lora_code2wav block (composite.code2wav_keys)

    target_modules: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for comp_name, rank, alpha, include_prefixes, exclude_prefixes in components:
        if not include_prefixes:
            continue
        count_before = len(target_modules)
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not _matches_prefix(name, include_prefixes):
                continue
            if exclude_prefixes and _matches_prefix(name, exclude_prefixes):
                continue
            target_modules.append(name)
            if rank != finetuning_args.lora_rank:
                rank_pattern[name] = rank
            if alpha != finetuning_args.lora_alpha:
                alpha_pattern[name] = alpha
        added = len(target_modules) - count_before
        if added == 0:
            logger.warning_rank0(
                f"lora_{comp_name} is enabled but no matching Linear modules found in the model. "
                f"Check that the model actually contains modules under prefixes: {include_prefixes}."
            )
        else:
            logger.info_rank0(f"lora_{comp_name}: {added} Linear modules selected (rank={rank}, alpha={alpha}).")

    if getattr(finetuning_args, "lora_language_model_experts", False):
        from .qwen3_omni_moe_lora import is_qwen3_omni_thinker_experts

        expert_rank = finetuning_args.lora_language_model_experts_rank
        expert_alpha = finetuning_args.lora_language_model_experts_alpha
        expert_count = 0
        for name, module in model.named_modules():
            if not is_qwen3_omni_thinker_experts(module):
                continue
            if not _matches_prefix(name, composite.language_model_keys):
                continue
            # Exclude Talker paths structurally (never target talker experts).
            if name.startswith("talker.") or ".talker." in name:
                continue
            target_modules.append(name)
            if expert_rank != finetuning_args.lora_rank:
                rank_pattern[name] = expert_rank
            if expert_alpha != finetuning_args.lora_alpha:
                alpha_pattern[name] = expert_alpha
            expert_count += 1
        if expert_count == 0:
            raise ValueError(
                "lora_language_model_experts is enabled but no Qwen3OmniMoeThinkerTextExperts "
                "modules were found under language_model prefixes. "
                "Fused Thinker experts (Transformers 5.x) are required."
            )
        logger.info_rank0(
            f"lora_language_model_experts: {expert_count} fused expert modules selected "
            f"(rank={expert_rank}, alpha={expert_alpha})."
        )

    return target_modules, rank_pattern, alpha_pattern


def _is_module_forbidden(name: str, freeze_modules: set[str], conflict_keys: set[str]) -> bool:
    """Check whether *name* should be excluded from LoRA/training.

    ``freeze_modules`` (vision tower, language model, talker, …) are matched
    by **prefix**: ``"model"`` matches ``model.layers.0.q_proj`` but NOT
    ``talker.model.layers.0.q_proj``.

    ``conflict_keys`` (e.g. ``patch_embed``) keep the original **substring**
    semantics because they denote module *types* that may appear at any depth.
    """
    for fm in freeze_modules:
        if name == fm or name.startswith(fm + "."):
            return True
    for ck in conflict_keys:
        if ck in name:
            return True
    return False


def patch_target_modules(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", target_modules: list[str]
) -> list[str]:
    r"""Freeze vision tower for VLM LoRA tuning."""
    model_type = getattr(model.config, "model_type", None)
    if model_type in COMPOSITE_MODELS:
        freeze_modules = get_forbidden_modules(model.config, finetuning_args)
        conflict_keys = set(COMPOSITE_MODELS[model_type].lora_conflict_keys)
        module_names = []
        for name, _ in model.named_modules():
            if any(target_module in name for target_module in target_modules) and not _is_module_forbidden(
                name, freeze_modules, conflict_keys
            ):
                module_names.append(name)

        return module_names
    else:
        return target_modules


_register_composite_model(
    model_type="dots_ocr",
    projector_key="vision_tower.merger",
    vision_model_keys=["vision_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["merger"],
)


_register_composite_model(
    model_type="gemma3",
)


_register_composite_model(
    model_type="gemma3n",
    vision_model_keys=["vision_tower", "audio_tower"],
    lora_conflict_keys=["timm_model", "subsample_conv_projection"],
)


# copied from qwen2vl
_register_composite_model(
    model_type="glm4v",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="glm4v_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="internvl",
)

_register_composite_model(
    model_type="interns1",
)

_register_composite_model(
    model_type="Keye",
    projector_key="mlp_AR",
    vision_model_keys=["visual.vision_model.patch_embedding", "visual.vision_model.encoder"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embedding"],
)


_register_composite_model(
    model_type="kimi_vl",
)


_register_composite_model(
    model_type="llama4",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="llava",
)


_register_composite_model(
    model_type="llava_next",
)


_register_composite_model(
    model_type="llava_next_video",
)


_register_composite_model(
    model_type="minicpmv",
    projector_key="resampler",
    vision_model_keys=["vpm"],
    language_model_keys=["llm"],
)


_register_composite_model(
    model_type="minicpmo",
    projector_key="resampler",
    vision_model_keys=["vpm", "apm", "audio_avg_pooler", "audio_projection_layer", "tts"],
    language_model_keys=["llm"],
    lora_conflict_keys=["audio_projection_layer"],
)


_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)


_register_composite_model(
    model_type="mllama",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="paligemma",
)


_register_composite_model(
    model_type="qwen2_audio",
    vision_model_keys=["audio_tower"],
)


_register_composite_model(
    model_type="qwen2_5_omni_thinker",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "audio_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["model.language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_5_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_omni_moe_thinker",
    projector_key="visual.merger",
    vision_model_keys=[
        "visual.pos_embed",
        "visual.patch_embed",
        "visual.blocks",
        "visual.deepstack_merger_list",
        "audio_tower",
    ],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="speechlmm",
    projector_key="visual.merger",
    vision_model_keys=[
        "visual.pos_embed",
        "visual.patch_embed",
        "visual.blocks",
        "visual.deepstack_merger_list",
    ],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
    audio_model_keys=["audio_tower"],
    # Qwen3-Omni: proj1/proj2/ln_post; Qwen2.5-Omni: proj/ln_post. Names that
    # the active backbone does not have never match anything.
    audio_adapter_prefixes=[
        "audio_tower.proj",
        "audio_tower.proj1",
        "audio_tower.proj2",
        "audio_tower.ln_post",
    ],
    lipread_model_keys=["lipread_encoder"],
    lipread_adapter_keys=["lipread_adapter"],
    talker_keys=["talker"],
    code2wav_keys=["code2wav"],
    # TODO: add talker_adapter_prefixes when lora_talker is implemented
    #   (need to decide how to handle talker.code_predictor.linear_heads ModuleList)
    # TODO: add vision_adapter_prefixes when lora_vision_encoder is implemented
)


_register_composite_model(
    model_type="video_llava",
)
