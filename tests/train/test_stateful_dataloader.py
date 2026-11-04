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

from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import DataLoader, Dataset

from llamafactory.train.stateful_dataloader import (
    build_stateful_data_config,
    load_dataloader_state,
    save_dataloader_state,
    tolerate_worker_state_snapshot,
)


class _Ids(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


def _data_config(**overrides) -> dict:
    values = {
        "dataset": ["toy"],
        "streaming": False,
        "mix_strategy": "concat",
        "interleave_probs": None,
        "buffer_size": 16384,
        "cutoff_len": 512,
        "packing": False,
        "neat_packing": False,
    }
    values.update(overrides)
    return build_stateful_data_config(SimpleNamespace(**values))


def _trainer(batch_size: int = 4):
    return SimpleNamespace(
        args=SimpleNamespace(
            process_index=0,
            world_size=1,
            dataloader_num_workers=0,
            gradient_accumulation_steps=1,
            seed=42,
            dataloader_drop_last=True,
        ),
        _train_batch_size=batch_size,
    )


def _loader(accelerator: Accelerator, batch_size: int = 4) -> DataLoader:
    return accelerator.prepare(
        DataLoader(
            _Ids(40),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=lambda batch: torch.tensor(batch),
        )
    )


def _batches(dataloader: DataLoader) -> list[list[int]]:
    return [batch.tolist() for batch in dataloader]


@pytest.fixture(scope="module")
def accelerator() -> Accelerator:
    return Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_stateful_dataloader=True,
            even_batches=False,
            dispatch_batches=False,
            use_seedable_sampler=False,
        )
    )


def test_round_trip_restores_the_unconsumed_tail(tmp_path, accelerator):
    trainer = _trainer()
    reference = _loader(accelerator)
    assert reference.use_stateful_dataloader
    tolerate_worker_state_snapshot(reference)

    seen = []
    for batch in reference:
        seen.append(batch.tolist())
        if len(seen) == 3:
            save_dataloader_state(trainer, reference, str(tmp_path), _data_config())

    restored = _loader(accelerator)
    load_dataloader_state(trainer, restored, str(tmp_path), _data_config())
    assert _batches(restored) == seen[3:]


def test_missing_state_file_raises(tmp_path, accelerator):
    trainer = _trainer()
    dataloader = _loader(accelerator)
    with pytest.raises(ValueError, match="was not found"):
        load_dataloader_state(trainer, dataloader, str(tmp_path), _data_config())


def test_batch_size_mismatch_raises(tmp_path, accelerator):
    trainer = _trainer(batch_size=4)
    dataloader = _loader(accelerator)
    next(iter(dataloader))
    save_dataloader_state(trainer, dataloader, str(tmp_path), _data_config())

    trainer._train_batch_size = 8
    restored = _loader(accelerator, batch_size=8)
    with pytest.raises(ValueError, match="batch_size"):
        load_dataloader_state(trainer, restored, str(tmp_path), _data_config())


def test_data_fingerprint_mismatch_raises(tmp_path, accelerator):
    trainer = _trainer()
    dataloader = _loader(accelerator)
    next(iter(dataloader))
    save_dataloader_state(trainer, dataloader, str(tmp_path), _data_config())

    restored = _loader(accelerator)
    with pytest.raises(ValueError, match="data configuration"):
        load_dataloader_state(trainer, restored, str(tmp_path), _data_config(dataset=["other"]))
