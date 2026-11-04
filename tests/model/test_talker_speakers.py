# Copyright 2025 the LlamaFactory team.
from types import SimpleNamespace

import pytest

from llamafactory.model.loader import _load_talker_speakers, _resolve_talker_speaker_bos


@pytest.mark.runs_on(["cpu", "mps"])
def test_resolve_talker_speaker_bos_sets_id():
    config = SimpleNamespace(
        talker_speaker="ethan",
        speaker_bos_token_id=None,
        resolve_talker_speaker_id=lambda name, mapping: mapping[name],
    )
    model = SimpleNamespace(config=config, speaker_map={"ethan": 7})
    _resolve_talker_speaker_bos(model)
    assert config.speaker_bos_token_id == 7


@pytest.mark.runs_on(["cpu", "mps"])
def test_resolve_talker_speaker_bos_swallows_unknown_speaker():
    config = SimpleNamespace(
        talker_speaker="missing",
        speaker_bos_token_id=3,
        resolve_talker_speaker_id=lambda name, mapping: (_ for _ in ()).throw(ValueError(name)),
    )
    model = SimpleNamespace(config=config, speaker_map={"ethan": 7})
    _resolve_talker_speaker_bos(model)
    assert config.speaker_bos_token_id == 3


@pytest.mark.runs_on(["cpu", "mps"])
def test_load_talker_speakers_loads_then_resolves(tmp_path):
    loaded = {}

    def load_speakers(path):
        loaded["path"] = path
        model.speaker_map = {"ethan": 11}

    config = SimpleNamespace(
        talker_speaker="ethan",
        speaker_bos_token_id=None,
        resolve_talker_speaker_id=lambda name, mapping: mapping[name],
    )
    model = SimpleNamespace(config=config, speaker_map=None, load_speakers=load_speakers)
    spk = tmp_path / "spk_dict.pt"
    spk.write_bytes(b"x")
    _load_talker_speakers(model, str(tmp_path))
    assert loaded["path"] == str(spk)
    assert config.speaker_bos_token_id == 11
