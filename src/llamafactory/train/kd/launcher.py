# Copyright 2025 Meetween / SpeechLMM KD extension.
"""Distributed launcher for KD training (paths with spaces safe via ``shlex``)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from copy import deepcopy


def launch() -> None:
    from ...extras import logging
    from ...extras.misc import find_available_port, get_device_count, is_env_enabled, use_kt, use_ray

    logger = logging.get_logger(__name__)
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command != "train":
        raise SystemExit("KD launcher supports only: python -m llamafactory.train.kd.launcher train <config.yaml> …")

    if not (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        from .cli import main

        main()
        return

    nnodes = os.getenv("NNODES", "1")
    node_rank = os.getenv("NODE_RANK", "0")
    nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))
    logger.info_rank0(f"KD torchrun: {nproc_per_node} processes @ {master_addr}:{master_port}")

    env = deepcopy(os.environ)
    if is_env_enabled("OPTIM_TORCH", "1"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    cmd = (
        f"torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
        f"--master_addr {master_addr} --master_port {master_port} -m llamafactory.train.kd {args}"
    )
    subprocess.run(shlex.split(cmd), env=env, check=True)


if __name__ == "__main__":
    launch()
