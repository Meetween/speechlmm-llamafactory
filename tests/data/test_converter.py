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

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_lipread": None,
        "_codec_tokens": None,
        "_speaker": None,
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_lipread": None,
        "_codec_tokens": None,
        "_speaker": None,
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_explicit_speaker_and_codec():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    dataset_attr.speaker = "spk"
    dataset_attr.codec_tokens = "codec_tokens"
    data_args = DataArguments()
    example = {
        "instruction": "Say hi.",
        "input": "",
        "output": "Hello.",
        "spk": "ethan",
        "speaker": "ignored",
        "codec_tokens": [1, 2, 3],
    }
    converted = get_dataset_converter("alpaca", dataset_attr, data_args)(example)
    assert converted["_speaker"] == "ethan"
    assert converted["_codec_tokens"] == [1, 2, 3]


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_implicit_speaker_ignored():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Say hi.",
        "input": "",
        "output": "Hello.",
        "speaker": "ethan",
    }
    converted = get_dataset_converter("alpaca", dataset_attr, data_args)(example)
    assert converted["_speaker"] is None
    assert converted["_codec_tokens"] is None


@pytest.mark.runs_on(["cpu", "mps"])
def test_join_codec_cache_defaults_sample_id():
    dataset_attr = DatasetAttr("file", "talker.json")
    dataset_attr.join(
        {
            "formatting": "sharegpt",
            "codec_cache": "/tmp/talker-codec-cache/qwen2_5_omni/qwen25-distill-v1",
            "columns": {"messages": "messages", "speaker": "speaker"},
        }
    )
    assert dataset_attr.codec_cache.endswith("qwen25-distill-v1")
    assert dataset_attr.sample_id == "sample_id"
    assert dataset_attr.speaker == "speaker"


@pytest.mark.runs_on(["cpu", "mps"])
def test_join_explicit_sample_id_column():
    dataset_attr = DatasetAttr("file", "talker.json")
    dataset_attr.join(
        {
            "formatting": "sharegpt",
            "codec_cache": "/tmp/cache",
            "columns": {"messages": "messages", "sample_id": "id"},
        }
    )
    assert dataset_attr.sample_id == "id"
