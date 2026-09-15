# Copyright 2025 the LlamaFactory team.
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

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from speechlmm.models.configuration_speechlmm import SpeechLMMConfig
from speechlmm.tokens import LIPREAD_BOS_TOKEN, LIPREAD_EOS_TOKEN, LIPREAD_PAD_TOKEN
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_torch_version_greater_than
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
TRAINABLE_MODULES_FILENAME = "trainable_modules.safetensors"


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _resolve_stage_base_delta(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / TRAINABLE_MODULES_FILENAME
    if not resolved.is_file():
        raise ValueError(f"Stage base delta does not exist: {resolved}")
    return resolved


def split_stage_base_deltas(value: str) -> list[str]:
    """Split a stage_base_delta value that may name several deltas.

    A stage can inherit the published modules of more than one earlier stage —
    for example an end-to-end stage continuing from an independently trained
    lipread adapter and audio adapter. Each entry is applied in order and the
    entries must not touch the same parameter.
    """
    return [entry.strip() for entry in str(value).split(",") if entry.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_parameter_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name.removeprefix("base_model.model.")
    return name.replace(".base_layer.", ".")


def load_stage_base_delta(model: torch.nn.Module, path: str) -> None:
    """Overlay strict full-weight deltas before PEFT/LoRA construction.

    ``path`` may name several deltas, comma separated; they are merged and must
    not overlap, so inheriting two independent adapter stages is unambiguous.
    """
    from safetensors.torch import load_file

    entries = split_stage_base_deltas(path)
    if not entries:
        raise ValueError(f"stage_base_delta is empty: {path!r}")
    delta_paths = [_resolve_stage_base_delta(entry) for entry in entries]

    state: dict[str, "torch.Tensor"] = {}
    provenance: dict[str, str] = {}
    for delta_path in delta_paths:
        loaded = dict(load_file(str(delta_path), device="cpu"))
        if not loaded:
            raise ValueError(f"Stage base delta is empty: {delta_path}")
        collisions = sorted(set(loaded) & set(state))
        if collisions:
            raise ValueError(
                f"Stage base deltas overlap on {collisions[:10]}: "
                f"{provenance[collisions[0]]} and {delta_path}"
            )
        state.update(loaded)
        provenance.update({key: str(delta_path) for key in loaded})
    if any("lora_" in key for key in state):
        raise ValueError("stage_base_delta must contain full weights, not LoRA tensors")

    named_parameters = dict(model.named_parameters())
    canonical_targets: dict[str, list[str]] = {}
    for name in named_parameters:
        canonical_targets.setdefault(_canonical_parameter_name(name), []).append(name)

    resolved_targets: dict[str, str] = {}
    errors: list[str] = []
    for key, tensor in state.items():
        candidates = canonical_targets.get(_canonical_parameter_name(key), [])
        if len(candidates) != 1:
            errors.append(f"{key}: matched {candidates}")
            continue
        target_name = candidates[0]
        target = named_parameters[target_name]
        target_shape = tuple(getattr(target, "ds_shape", target.shape))
        if target_shape != tuple(tensor.shape):
            errors.append(f"{key}: checkpoint shape={tuple(tensor.shape)}, model shape={target_shape}")
            continue
        resolved_targets[key] = target_name
    if errors:
        raise ValueError("Stage base delta does not match the running model:\n- " + "\n- ".join(errors[:20]))

    zero3 = is_deepspeed_zero3_enabled()
    rank = (
        torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    )
    for key, target_name in resolved_targets.items():
        parameter = named_parameters[target_name]
        if zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
                if rank == 0:
                    parameter.data.copy_(
                        state[key].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
        else:
            parameter.data.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))

    model._speechlmm_stage_base_delta = {
        "paths": [str(delta_path.resolve()) for delta_path in delta_paths],
        "sha256": [_file_sha256(delta_path) for delta_path in delta_paths],
        "keys": sorted(state),
    }
    logger.info_rank0(
        f"Loaded {len(state)} inherited full-weight tensors from "
        + ", ".join(str(delta_path) for delta_path in delta_paths)
    )


@contextmanager
def _disable_zero3_pointer_tie_inference(model_class: type["PreTrainedModel"]):
    """Avoid treating ZeRO-3 placeholder pointers as real tied parameters.

    During ``zero.Init`` every partitioned parameter can temporarily expose the
    same zero-sized storage pointer. Transformers' compatibility fallback then
    mistakes all of those parameters for one tied-weight group. Explicit ties
    are already present in ``all_tied_weights_keys``; only the unreliable
    pointer-discovery fallback is suppressed while loading the Thinker.
    """
    if not is_deepspeed_zero3_enabled():
        yield
        return

    original = model_class._adjust_tied_keys_with_tied_pointers
    model_class._adjust_tied_keys_with_tied_pointers = lambda self, missing_keys: None
    try:
        yield
    finally:
        model_class._adjust_tied_keys_with_tied_pointers = original


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def _ensure_speechlmm_registered(model_path: str) -> None:
    r"""Register SpeechLMM with HuggingFace Auto classes before AutoConfig.load."""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        return
    with open(config_file, encoding="utf-8") as f:
        model_type = json.load(f).get("model_type")
    if model_type == "speechlmm":
        import speechlmm.models  # noqa: F401


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    _ensure_speechlmm_registered(model_args.model_name_or_path)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable, finetuning_args=finetuning_args)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        elif getattr(config, "model_type", None) == "speechlmm":
            from speechlmm.models import SpeechLMMForConditionalGeneration

            if finetuning_args.freeze_talker:
                config.enable_talker = False
            if finetuning_args.freeze_code2wav:
                config.enable_code2wav = False

            if model_args.train_from_scratch:
                model = SpeechLMMForConditionalGeneration._from_config(config)
            else:
                model = SpeechLMMForConditionalGeneration.from_pretrained(**init_kwargs)

        elif model_args.use_speechlmm_wrapper and getattr(config, "model_type", None) in (
            "qwen2_5_omni",
            "qwen3_omni_moe",
        ):
            from speechlmm.models import SpeechLMMForConditionalGeneration

            config_overrides = {}
            has_trainable_adapters = bool(finetuning_args.trainable_module_paths)
            has_trainable_lipread = (
                not finetuning_args.freeze_lipread_encoder or not finetuning_args.freeze_lipread_adapter
            )
            if finetuning_args.freeze_language_model and not has_trainable_adapters and not has_trainable_lipread:
                config_overrides["thinker_loss_weight"] = 0.0

            config_overrides["enable_lipread"] = has_trainable_lipread
            config_overrides["enable_talker"] = not finetuning_args.freeze_talker
            config_overrides["enable_code2wav"] = not finetuning_args.freeze_code2wav
            config_overrides["lipread_bos_token_id"] = tokenizer.vocab[LIPREAD_BOS_TOKEN]
            config_overrides["lipread_eos_token_id"] = tokenizer.vocab[LIPREAD_EOS_TOKEN]
            config_overrides["lipread_pad_token_id"] = tokenizer.vocab[LIPREAD_PAD_TOKEN]
            lipread_encoder_weights = getattr(model_args, "lipread_encoder_weights", None)
            if lipread_encoder_weights is not None:
                config_overrides["lipread_encoder_weights"] = lipread_encoder_weights

            speechlmm_config = SpeechLMMConfig.from_qwen_omni_config(config, **config_overrides)
            if (
                getattr(config, "model_type", None) == "qwen3_omni_moe"
                and finetuning_args.freeze_talker
                and finetuning_args.freeze_code2wav
            ):
                from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                    Qwen3OmniMoeThinkerForConditionalGeneration,
                )

                thinker_kwargs = dict(init_kwargs)
                thinker_config = config.thinker_config
                # The composite Thinker config inherits Transformers' generic
                # default (True), while its authoritative text config and the
                # checkpoint both use an independent lm_head.  Under ZeRO-3,
                # leaving the outer default enabled makes storage-based tied-
                # weight detection incorrectly associate the first sharded
                # audio parameter with every Thinker tensor.
                thinker_config.tie_word_embeddings = bool(
                    getattr(thinker_config.text_config, "tie_word_embeddings", False)
                )
                thinker_kwargs["config"] = thinker_config
                with _disable_zero3_pointer_tie_inference(Qwen3OmniMoeThinkerForConditionalGeneration):
                    qwen3_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(**thinker_kwargs)
            elif (
                getattr(config, "model_type", None) == "qwen2_5_omni"
                and finetuning_args.freeze_talker
                and finetuning_args.freeze_code2wav
            ):
                # Same reasoning as the Qwen3-Omni branch above: load only the
                # Thinker so the frozen Talker and Token2Wav weights are never
                # materialized, which matters most under ZeRO-3.
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                    Qwen2_5OmniThinkerForConditionalGeneration,
                )

                thinker_kwargs = dict(init_kwargs)
                thinker_config = config.thinker_config
                thinker_config.tie_word_embeddings = bool(
                    getattr(thinker_config.text_config, "tie_word_embeddings", False)
                )
                thinker_kwargs["config"] = thinker_config
                with _disable_zero3_pointer_tie_inference(
                    Qwen2_5OmniThinkerForConditionalGeneration
                ):
                    qwen3_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        **thinker_kwargs
                    )
            else:
                qwen3_model = AutoModelForTextToWaveform.from_pretrained(**init_kwargs)
            model = SpeechLMMForConditionalGeneration._wrap_qwen_omni(qwen3_model, speechlmm_config)
            if lipread_encoder_weights is not None:
                model.load_auto_avsr_weights(lipread_encoder_weights)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        if getattr(model.config, "model_type", None) == "speechlmm":
            SpeechLMMConfig.sync_lipread_token_ids_from_tokenizer(model.config, tokenizer)

    if model_args.stage_base_delta is not None:
        if not is_trainable:
            raise ValueError("stage_base_delta is only supported when starting a training stage")
        load_stage_base_delta(model, model_args.stage_base_delta)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # The AutoAVSR lipread encoder loses its BatchNorm running stats on load: under
    # ZeRO-3 the 0-sized init makes transformers' storage-based tied-weight detection group
    # every lipread param/buffer, so the BN buffers get dropped by from_pretrained (speechlmm
    # resume) or reset by PEFT (wrapper path). Reloading the pristine .pth restores them
    # exactly. load_auto_avsr_weights is ZeRO-3-safe (GatheredParameters) and maps original
    # Linear keys onto PEFT base_layer keys when lipreading LoRA is active. Requires
    # lipread_encoder_weights to be set on the resume config.
    lipread_weights = getattr(model_args, "lipread_encoder_weights", None)
    if lipread_weights and hasattr(model, "load_auto_avsr_weights"):
        model.load_auto_avsr_weights(lipread_weights)
        logger.info_rank0(f"Restored AutoAVSR lipread encoder from {lipread_weights}")

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # Conv3D is not recommended when using torch 2.9.x
    if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
        if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
            raise ValueError(
                "Unsupported torch version detected: torch 2.9.x with Conv3D. "
                "This combination is known to cause severe performance regression. "
                "Please downgrade torch to <2.9 or remove Conv3D. "
                "See https://github.com/pytorch/pytorch/issues/166122"
            )

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model, include_kernels=model_args.use_v1_kernels)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
