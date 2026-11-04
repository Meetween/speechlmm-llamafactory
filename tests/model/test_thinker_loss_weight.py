# Copyright 2025 the LlamaFactory team.
from types import SimpleNamespace

import pytest

from llamafactory.model.loader import thinker_is_fully_frozen


def _args(**overrides):
    values = dict(
        freeze_language_model=True,
        trainable_module_paths=None,
        freeze_lipread_encoder=True,
        freeze_lipread_adapter=True,
        freeze_audio_tower=True,
        freeze_audio_encoder=True,
        freeze_audio_adapters=True,
        lora_audio_encoder=False,
        freeze_multi_modal_projector=True,
        freeze_vision_tower=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.runs_on(["cpu", "mps"])
def test_thinker_frozen_when_language_and_towers_are_frozen():
    assert thinker_is_fully_frozen(_args()) is True


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"freeze_language_model": False},
        {"trainable_module_paths": ["audio_tower.proj1"]},
        {"lora_audio_encoder": True},
        {"freeze_audio_tower": False},
        {"freeze_audio_encoder": False},
        {"freeze_audio_adapters": False},
        {"freeze_multi_modal_projector": False},
        {"freeze_vision_tower": False},
        {"freeze_lipread_encoder": False},
        {"freeze_lipread_adapter": False},
    ],
)
def test_thinker_not_frozen_when_a_thinker_module_trains(overrides):
    assert thinker_is_fully_frozen(_args(**overrides)) is False
