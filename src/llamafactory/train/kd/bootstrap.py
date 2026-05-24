# Copyright 2025 Meetween / SpeechLMM KD extension.
"""Register KD with LlamaFactory at runtime (no edits outside ``train/kd``)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .hparams import FinetuningArgumentsKD


if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from ...hparams import ModelArguments


_INTEGRATED = False


def _patch_finetuning_parser() -> None:
    from ...hparams import parser as parser_mod

    parser_mod.FinetuningArguments = FinetuningArgumentsKD  # type: ignore[misc, assignment]
    parser_mod._TRAIN_ARGS = [
        parser_mod.ModelArguments,
        parser_mod.DataArguments,
        parser_mod.TrainingArguments,
        FinetuningArgumentsKD,
        parser_mod.GeneratingArguments,
    ]
    parser_mod._TRAIN_CLS = tuple(parser_mod._TRAIN_ARGS)
    if parser_mod._TRAIN_MCA_ARGS:
        parser_mod._TRAIN_MCA_ARGS = [
            parser_mod.ModelArguments,
            parser_mod.DataArguments,
            parser_mod._TRAIN_MCA_ARGS[2],
            FinetuningArgumentsKD,
            parser_mod.GeneratingArguments,
        ]
        parser_mod._TRAIN_MCA_CLS = tuple(parser_mod._TRAIN_MCA_ARGS)


def _patch_load_config() -> None:
    from transformers import AutoConfig

    from ...extras import logging
    from ...model import loader as loader_mod

    logger = logging.get_logger(__name__)
    _orig = loader_mod.load_config

    @functools.wraps(_orig)
    def load_config(model_args: "ModelArguments") -> "PreTrainedConfig":
        init_kwargs = loader_mod._get_init_kwargs(model_args)
        try:
            return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        except ValueError as exc:
            if "speechlmm" not in str(exc).lower():
                raise
            from speechlmm.models.configuration_speechlmm import SpeechLMMConfig

            logger.info_rank0(
                "Checkpoint uses `model_type: speechlmm`; loading config with `SpeechLMMConfig`."
            )
            return SpeechLMMConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)

    loader_mod.load_config = load_config


def _patch_model_args_tokens() -> None:
    from ...hparams import model_args as model_args_mod

    _orig_post = model_args_mod.ModelArguments.__post_init__

    @functools.wraps(_orig_post)
    def post_init(self: model_args_mod.ModelArguments) -> None:
        tok = getattr(self, "add_special_tokens", None)
        if isinstance(tok, (list, tuple)):
            # SpeechLMM workflow passes a list; BaseModelArguments.__post_init__ expects str.
            normalized = [str(t).strip() for t in tok if str(t).strip()]
            self.add_special_tokens = ",".join(normalized) if normalized else None
        elif tok is not None and not isinstance(tok, str):
            raise ValueError(f"`add_special_tokens` must be str or list, got {type(tok)}")
        _orig_post(self)

    model_args_mod.ModelArguments.__post_init__ = post_init  # type: ignore[method-assign]


def _patch_training_dispatch() -> None:
    from ...train import tuner as tuner_mod

    _orig = tuner_mod._training_function

    @functools.wraps(_orig)
    def _training_function(config: dict[str, Any]) -> None:
        from ...hparams import get_train_args

        args = config.get("args")
        callbacks: list[Any] = config.get("callbacks") or []
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

        if finetuning_args.stage == "kd":
            from .workflow import run_kd

            run_kd(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
            if tuner_mod.is_ray_available() and tuner_mod.ray.is_initialized():
                return
            try:
                if tuner_mod.dist.is_initialized():
                    tuner_mod.dist.destroy_process_group()
            except Exception as exc:
                tuner_mod.logger.warning(f"Failed to destroy process group: {exc}.")
            return

        return _orig(config)

    tuner_mod._training_function = _training_function


def _patch_load_auto_avsr_weights() -> None:
    """Skip ``load_auto_avsr_weights(None)`` when the teacher is a full checkpoint.

    ``loader.load_model`` always calls this after wrapping Qwen3-Omni; lipread weights
    are already in the teacher shards (or lipread is disabled for frozen ref models).
    """
    try:
        from speechlmm.models import SpeechLMMForConditionalGeneration
    except ImportError:
        return

    _orig = SpeechLMMForConditionalGeneration.load_auto_avsr_weights

    @functools.wraps(_orig)
    def load_auto_avsr_weights(self, autoavsr_weights_path: str | None) -> None:
        if autoavsr_weights_path is None or not str(autoavsr_weights_path).strip():
            return
        return _orig(self, autoavsr_weights_path)

    SpeechLMMForConditionalGeneration.load_auto_avsr_weights = load_auto_avsr_weights  # type: ignore[method-assign]
    if hasattr(SpeechLMMForConditionalGeneration, "load_autoAVSR_weights"):
        SpeechLMMForConditionalGeneration.load_autoAVSR_weights = load_auto_avsr_weights  # type: ignore[method-assign]


def _patch_eval_warnings() -> None:
    from ...extras import logging
    from ...hparams import parser as parser_mod

    logger = logging.get_logger(__name__)
    _orig = parser_mod.get_train_args

    @functools.wraps(_orig)
    def get_train_args(args=None):
        result = _orig(args)
        model_args, data_args, training_args, finetuning_args, generating_args = result
        if (
            not training_args.do_train
            and finetuning_args.stage == "kd"
            and finetuning_args.ref_model is None
        ):
            logger.warning_rank0("Specify `ref_model` for KD when evaluating (teacher logits / alignment).")
        return result

    parser_mod.get_train_args = get_train_args


def integrate() -> None:
    """Apply KD hooks once per process (safe to call multiple times)."""
    global _INTEGRATED
    if _INTEGRATED:
        return
    _patch_finetuning_parser()
    _patch_load_config()
    _patch_load_auto_avsr_weights()
    _patch_model_args_tokens()
    _patch_training_dispatch()
    _patch_eval_warnings()
    _INTEGRATED = True
