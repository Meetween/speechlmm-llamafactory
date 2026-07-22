# Copyright 2025 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from speechlmm.tokens import LIPREAD_FPS, LIPREAD_FRAME_SIZE, LIPREAD_PLACEHOLDER

from llamafactory.data.mm_plugin import LipreadTokenLayout, SpeechLMMPlugin


def test_speechlmm_lipread_layout_emits_bos_pad_eos_thinker_tokens():
    plugin = SpeechLMMPlugin(image_token="<image>", video_token="<video>", audio_token="<audio>")
    frames = 25
    fake_tensor = torch.zeros(frames, LIPREAD_FRAME_SIZE, LIPREAD_FRAME_SIZE)
    messages = [{"role": "user", "content": f"{LIPREAD_PLACEHOLDER}Read my lips."}]
    processor = SimpleNamespace()

    with (
        patch.object(SpeechLMMPlugin, "_validate_input", return_value=None),
        patch.object(
            SpeechLMMPlugin,
            "_get_mm_inputs",
            return_value={"lipread": [fake_tensor]},
        ),
        patch(
            "llamafactory.data.mm_plugin.Qwen2OmniPlugin.process_messages_with_layout",
            return_value=(messages, SimpleNamespace(audios=(), images=(), videos=())),
        ),
    ):
        processed, layouts = plugin.process_messages_with_layout(
            messages,
            images=[],
            videos=[],
            audios=[],
            processor=processor,
            lipread=["dummy.mp4"],
        )

    assert len(layouts.lipreads) == 1
    layout = layouts.lipreads[0]
    assert isinstance(layout, LipreadTokenLayout)
    assert layout.frames == frames
    assert layout.height == LIPREAD_FRAME_SIZE
    assert layout.width == LIPREAD_FRAME_SIZE
    assert layout.channels == 1
    assert layout.thinker_tokens == frames + 2
    assert layout.source_duration_seconds == frames / float(LIPREAD_FPS)
    assert not layout.truncated
    assert plugin.lipread_bos_token in processed[0]["content"]
    assert plugin.lipread_eos_token in processed[0]["content"]
    assert LIPREAD_PLACEHOLDER not in processed[0]["content"]
