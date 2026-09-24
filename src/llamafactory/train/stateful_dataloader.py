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
from typing import TYPE_CHECKING

import torch

from ..extras import logging


if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import Trainer

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)

DATALOADER_STATE_NAME = "dataloader_state_rank_{}.pt"
_STATE_VERSION = 1
_DATA_CONFIG_KEYS = (
    "dataset",
    "streaming",
    "mix_strategy",
    "interleave_probs",
    "buffer_size",
    "cutoff_len",
    "packing",
    "neat_packing",
)


def build_stateful_data_config(data_args: "DataArguments") -> dict:
    return {key: getattr(data_args, key) for key in _DATA_CONFIG_KEYS}


def data_config_fingerprint(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def tolerate_worker_state_snapshot(dataloader: "DataLoader") -> None:
    """Skip accelerate's prefetch rewind when the state is a worker snapshot.

    ``DataLoaderAdapter.adjust_state_dict_for_prefetch`` always reads
    ``_sampler_iter_yielded``. torchdata only puts that key at the top level when
    ``num_workers == 0``. With workers, the call raises ``KeyError`` on every batch.
    """
    original = dataloader.adjust_state_dict_for_prefetch

    def adjust() -> None:
        state = getattr(dataloader, "dl_state_dict", None)
        if isinstance(state, dict) and "_sampler_iter_yielded" not in state:
            return
        original()

    dataloader.adjust_state_dict_for_prefetch = adjust


def enable_stateful_dataloader(accelerator: "Accelerator") -> None:
    from accelerate.utils import is_torchdata_stateful_dataloader_available

    if not is_torchdata_stateful_dataloader_available():
        raise ImportError("`use_stateful_dataloader` requires torchdata with StatefulDataLoader.")

    if not hasattr(accelerator.dataloader_config, "use_stateful_dataloader"):
        raise ImportError(
            "`use_stateful_dataloader` requires an accelerate version with "
            "`DataLoaderConfiguration.use_stateful_dataloader`."
        )

    accelerator.dataloader_config.use_stateful_dataloader = True


def _state_path(directory: str, process_index: int) -> str:
    return os.path.join(directory, DATALOADER_STATE_NAME.format(process_index))


def _envelope(trainer: "Trainer") -> dict:
    return {
        "rank": trainer.args.process_index,
        "world_size": trainer.args.world_size,
        "num_workers": trainer.args.dataloader_num_workers,
        "batch_size": trainer._train_batch_size,
        "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        "seed": trainer.args.seed,
        "drop_last": trainer.args.dataloader_drop_last,
    }


def save_dataloader_state(trainer: "Trainer", dataloader: "DataLoader", output_dir: str, data_config: dict) -> None:
    state = {
        "version": _STATE_VERSION,
        "dataloader": dataloader.state_dict(),
        **_envelope(trainer),
        "data_config": data_config,
        "data_fingerprint": data_config_fingerprint(data_config),
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, _state_path(output_dir, trainer.args.process_index))


def _yielded_count(state: dict):
    if state.get("_num_yielded") is not None:
        return state["_num_yielded"]
    snapshot = state.get("_snapshot")
    if isinstance(snapshot, dict):
        return snapshot.get("_snapshot_step")
    return None


def load_dataloader_state(
    trainer: "Trainer", dataloader: "DataLoader", checkpoint_dir: str, data_config: dict
) -> None:
    state_file = _state_path(checkpoint_dir, trainer.args.process_index)
    if not os.path.isfile(state_file):
        raise ValueError(
            f"Stateful dataloader state for rank {trainer.args.process_index} was not found in {checkpoint_dir}."
        )

    state = torch.load(state_file, map_location="cpu", weights_only=False)
    version = state.get("version", 1)
    if version != _STATE_VERSION:
        raise ValueError(f"Unsupported stateful dataloader checkpoint version: {version}.")

    for key, value in _envelope(trainer).items():
        if state.get(key) != value:
            raise ValueError(
                f"Stateful dataloader checkpoint is incompatible: {key}={state.get(key)} in checkpoint, "
                f"current value={value}."
            )

    if state.get("data_fingerprint") != data_config_fingerprint(data_config):
        raise ValueError(
            "Stateful dataloader checkpoint is incompatible with the current data configuration.\n"
            f"Checkpoint: {state.get('data_config')}\n"
            f"Current: {data_config}"
        )

    dataloader.load_state_dict(state["dataloader"])
    yielded = _yielded_count(state["dataloader"])
    logger.info_rank0(f"Restored training dataloader state from {state_file} (yielded={yielded}).")
