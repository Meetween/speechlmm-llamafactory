# Copyright 2025 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from speechlmm.tokens import LIPREAD_FPS, LIPREAD_FRAME_SIZE, LIPREAD_PLACEHOLDER

from llamafactory.data.mm_plugin import LipreadProcessor, LipreadTokenLayout, SpeechLMMPlugin


def test_speechlmm_lipread_layout_emits_bos_pad_eos_thinker_tokens():
    plugin = SpeechLMMPlugin(image_token="<image>", video_token="<video>", audio_token="<audio>")
    frames = 25
    messages = [{"role": "user", "content": f"{LIPREAD_PLACEHOLDER}Read my lips."}]
    processor = SimpleNamespace()

    with (
        patch.object(SpeechLMMPlugin, "_validate_input", return_value=None),
        patch.object(LipreadProcessor, "count_frames", return_value=frames),
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


def test_speechlmm_qwen2_omni_template_uses_qwen25_multimodal_token_names():
    """The Qwen2.5-Omni vocabulary has no <|audio_pad|>/<|audio_start|> tokens.

    Emitting the Qwen3 names would tokenize them as plain text, inflating every
    audio span and leaving masked_scatter with zero placeholder positions.
    """
    from llamafactory.data.template import TEMPLATES

    template = TEMPLATES["speechlmm_qwen2_omni"]
    plugin = template.mm_plugin

    assert isinstance(plugin, SpeechLMMPlugin)
    assert (plugin.image_token, plugin.video_token, plugin.audio_token) == (
        "<|IMAGE|>",
        "<|VIDEO|>",
        "<|AUDIO|>",
    )
    assert (plugin.audio_bos_token, plugin.audio_eos_token) == ("<|audio_bos|>", "<|audio_eos|>")
    assert (plugin.vision_bos_token, plugin.vision_eos_token) == ("<|vision_bos|>", "<|vision_eos|>")


def test_speechlmm_qwen2_omni_template_matches_qwen2_omni_for_non_lipread_data():
    """Only the plugin may differ, so already-prepared audio corpora stay valid."""
    from llamafactory.data.template import TEMPLATES

    speechlmm = TEMPLATES["speechlmm_qwen2_omni"]
    qwen2_omni = TEMPLATES["qwen2_omni"]

    for field in ("format_user", "format_assistant", "format_system", "format_observation"):
        assert getattr(speechlmm, field).slots == getattr(qwen2_omni, field).slots, field
    assert speechlmm.default_system == qwen2_omni.default_system
    assert speechlmm.stop_words == qwen2_omni.stop_words
    assert speechlmm.replace_eos == qwen2_omni.replace_eos
    assert type(speechlmm) is type(qwen2_omni)


def test_is_speechlmm_template_covers_both_backbones():
    from speechlmm.tokens import is_speechlmm_template

    assert is_speechlmm_template("speechlmm")
    assert is_speechlmm_template("speechlmm_qwen2_omni")
    assert not is_speechlmm_template("qwen2_omni")
    assert not is_speechlmm_template(None)


class _FakeMetadata:
    def __init__(self, begin, end, average_fps):
        self.begin_stream_seconds = begin
        self.end_stream_seconds = end
        self.average_fps = average_fps


class _FakeDecoder:
    def __init__(self, metadata, num_frames=100):
        self.metadata = metadata
        self._num_frames = num_frames

    def __len__(self):
        return self._num_frames


def _count_with_fake_decoder(begin, end, average_fps=29.97, num_frames=100):
    from llamafactory.data.mm_plugin import LipreadProcessor

    decoder = _FakeDecoder(_FakeMetadata(begin, end, average_fps), num_frames)
    fake_module = SimpleNamespace(VideoDecoder=lambda path: decoder)
    with patch.dict("sys.modules", {"torchcodec.decoders": fake_module}):
        return LipreadProcessor().count_frames("dummy.mp4")


def test_count_frames_matches_the_sampler_arange():
    """clips_at_regular_timestamps emits one clip per arange step over the stream."""
    for begin, end in [(0.0, 7.24), (0.0, 3.84), (0.0, 1.0), (0.5, 10.34), (0.0, 0.01)]:
        expected = torch.arange(begin, end, 1 / LIPREAD_FPS)
        if expected[-1] >= end:
            expected = expected[expected < end]
        assert _count_with_fake_decoder(begin, end) == expected.numel(), (begin, end)


def test_count_frames_excludes_a_clip_starting_exactly_at_the_stream_end():
    """sampling_range_end is exclusive, so a step landing on it must not count."""
    # 25 steps of 1/25 reach exactly 1.0, which is the exclusive upper bound.
    assert _count_with_fake_decoder(0.0, 1.0) == 25


def test_count_frames_falls_back_to_decoding_when_metadata_is_unusable():
    """Unusable metadata must reach the sampler so it raises its own error."""
    from llamafactory.data.mm_plugin import LipreadProcessor

    decoder = _FakeDecoder(_FakeMetadata(0.0, None, None))
    fake_module = SimpleNamespace(VideoDecoder=lambda path: decoder)
    with (
        patch.dict("sys.modules", {"torchcodec.decoders": fake_module}),
        patch.object(LipreadProcessor, "__call__", return_value=torch.zeros(42, 88, 88)) as decoded,
    ):
        assert LipreadProcessor().count_frames("dummy.mp4") == 42
    decoded.assert_called_once()


def test_lipread_layout_expansion_never_decodes_frames():
    """Placeholder sizing must stay off the decode path."""
    plugin = SpeechLMMPlugin(image_token="<image>", video_token="<video>", audio_token="<audio>")
    messages = [{"role": "user", "content": f"{LIPREAD_PLACEHOLDER}Read my lips."}]

    with (
        patch.object(SpeechLMMPlugin, "_validate_input", return_value=None),
        patch.object(SpeechLMMPlugin, "_get_mm_inputs") as get_mm_inputs,
        patch.object(LipreadProcessor, "count_frames", return_value=31) as count_frames,
        patch.object(LipreadProcessor, "__call__") as decode,
        patch(
            "llamafactory.data.mm_plugin.Qwen2OmniPlugin.process_messages_with_layout",
            return_value=(messages, SimpleNamespace(audios=(), images=(), videos=())),
        ),
    ):
        _processed, layouts = plugin.process_messages_with_layout(
            messages, images=[], videos=[], audios=[], processor=SimpleNamespace(), lipread=["dummy.mp4"]
        )

    assert layouts.lipreads[0].frames == 31
    assert layouts.lipreads[0].thinker_tokens == 33
    count_frames.assert_called_once_with("dummy.mp4")
    decode.assert_not_called()
    get_mm_inputs.assert_not_called()
