"""Regression: heterogeneous media layouts under multi-worker Arrow maps."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from datasets import Dataset
from speechlmm.data_loading_optimization.sample_shapes import (
    IMAGE_LAYOUTS_COLUMN,
    PREPARED_TOKENIZED_FEATURES,
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
    VIDEO_LAYOUTS_COLUMN,
    VISUAL_LAYOUT_FEATURE,
)

from llamafactory.data.loader import _get_preprocessed_dataset
from llamafactory.data.mm_plugin import MultimodalTokenLayouts, VisualTokenLayout
from llamafactory.data.preparation_errors import PreprocessedDatasetResult
from llamafactory.hparams import DataArguments


def _visual(modality: str, thinker_tokens: int) -> VisualTokenLayout:
    return VisualTokenLayout(
        modality=modality,
        grid_t=1 if modality == "image" else 2,
        grid_h=4,
        grid_w=4,
        merge_size=2,
        thinker_tokens=thinker_tokens,
        sampled_frames=1 if modality == "image" else 2,
        seconds_per_grid=0.0 if modality == "image" else 0.5,
        source_duration_seconds=None if modality == "image" else 1.0,
        truncated=False,
    )


class HeterogeneousMMPlugin:
    def process_messages_with_layout(self, messages, images, videos, audios, processor):
        layouts = MultimodalTokenLayouts(
            images=tuple(_visual("image", 16) for _ in (images or [])),
            videos=tuple(_visual("video", 32) for _ in (videos or [])),
        )
        return messages, layouts

    def process_messages(self, messages, images, videos, audios, processor):
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        return input_ids, labels


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens=False):
        return str(token_ids)


class FakeTemplate:
    efficient_eos = False
    mm_plugin = HeterogeneousMMPlugin()

    def encode_multiturn_batch(self, tokenizer, batch_messages, systems, tools):
        return [[([11], [12])] for _ in batch_messages]

    def encode_multiturn(self, tokenizer, messages, system, tools):
        return [([11], [12])]


def _heterogeneous_source(row_count: int = 64) -> Dataset:
    prompts = []
    images = []
    videos = []
    for index in range(row_count):
        group = index % 4
        if group == 0:
            prompts.append("image-only")
            images.append(["img.jpg"])
            videos.append(None)
        elif group == 1:
            prompts.append("video-only")
            images.append(None)
            videos.append(["vid.mp4"])
        elif group == 2:
            prompts.append("mixed")
            images.append(["img.jpg"])
            videos.append(["vid.mp4"])
        else:
            prompts.append("text-only")
            images.append(None)
            videos.append(None)
    size = len(prompts)
    return Dataset.from_dict(
        {
            "_prompt": [[{"role": "user", "content": content}] for content in prompts],
            "_response": [[{"role": "assistant", "content": "answer"}] for _ in prompts],
            "_system": [""] * size,
            "_tools": [""] * size,
            "_images": images,
            "_videos": videos,
            "_audios": [None] * size,
            "_codec_tokens": [None] * size,
            SOURCE_ID_COLUMN: ["source"] * size,
            SAMPLE_ID_COLUMN: [f"source:{index}" for index in range(size)],
            "_sample_shape_alignment_error": [None] * size,
        }
    )


def _data_args(tmp_path, *, workers: int):
    return DataArguments(
        build_sample_shape_index=True,
        tokenized_path=str(tmp_path / "tokenized"),
        preprocessing_num_workers=workers,
        preprocessing_batch_size=8,
    )


def _training_args():
    return SimpleNamespace(predict_with_generate=False, local_process_index=0, should_log=False)


@pytest.mark.parametrize("workers", [4])
def test_prepared_parallel_map_preserves_visual_layout_schema(tmp_path, workers):
    source = _heterogeneous_source(64)
    result = _get_preprocessed_dataset(
        source,
        _data_args(tmp_path, workers=workers),
        _training_args(),
        "sft",
        FakeTemplate(),
        FakeTokenizer(),
        return_result=True,
    )

    assert isinstance(result, PreprocessedDatasetResult)
    assert len(result.dataset) == 64
    assert result.dataset[SAMPLE_ID_COLUMN] == [f"source:{index}" for index in range(64)]
    assert result.dataset.features[IMAGE_LAYOUTS_COLUMN] == VISUAL_LAYOUT_FEATURE
    assert result.dataset.features[VIDEO_LAYOUTS_COLUMN] == VISUAL_LAYOUT_FEATURE
    assert result.dataset.features[IMAGE_LAYOUTS_COLUMN] == PREPARED_TOKENIZED_FEATURES[IMAGE_LAYOUTS_COLUMN]
    assert result.dataset.features[VIDEO_LAYOUTS_COLUMN] == PREPARED_TOKENIZED_FEATURES[VIDEO_LAYOUTS_COLUMN]

    for index, row in enumerate(result.dataset):
        group = index % 4
        image_layouts = row[IMAGE_LAYOUTS_COLUMN]
        video_layouts = row[VIDEO_LAYOUTS_COLUMN]
        if group in (0, 2):
            assert len(image_layouts) == 1
            assert image_layouts[0]["modality"] == "image"
            assert image_layouts[0]["thinker_tokens"] == 16
        else:
            assert image_layouts == []
        if group in (1, 2):
            assert len(video_layouts) == 1
            assert video_layouts[0]["modality"] == "video"
            assert video_layouts[0]["thinker_tokens"] == 32
        else:
            assert video_layouts == []
        # Null-only neighbors must still use list types, never list<null>.
        assert isinstance(image_layouts, list)
        assert isinstance(video_layouts, list)
        assert isinstance(row["images"], list)
        assert isinstance(row["videos"], list)
