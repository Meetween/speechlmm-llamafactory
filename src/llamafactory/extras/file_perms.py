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

r"""Make the files written by safetensors honour the process umask.

safetensors serialises through a ``tempfile::NamedTempFile`` -- created 0600, with the umask deliberately
bypassed -- and ``rename()``s it into place. The mode travels with the inode, so every ``*.safetensors`` lands
at 0600 while the rest of a checkpoint lands at 0666 & ~umask. On a shared filesystem that locks the group out
of the weights:

    -rw-rw---- adapter_config.json
    -rw------- adapter_model.safetensors
    -rw-rw---- trainer_state.json
    -rw------- trainable_modules.safetensors

The writers are spread across transformers, peft, accelerate and our own callbacks, and they all bind
``safetensors.torch.save_file`` at import time, so the single fix is to wrap that function and rebind the
aliases the already-imported modules are holding.
"""

import functools
import os
import sys


_installed = False


def _umask_mode() -> int:
    r"""Return the mode a plain ``open()`` would have produced, i.e. 0666 & ~umask."""
    umask = os.umask(0o077)  # read-modify-write; no portable way to query the umask
    os.umask(umask)
    return 0o666 & ~umask


def install_safetensors_umask() -> None:
    r"""Chmod safetensors output to 0666 & ~umask. Idempotent; call once, at process start."""
    global _installed
    if _installed:
        return

    import safetensors.torch

    original = safetensors.torch.save_file
    mode = _umask_mode()

    @functools.wraps(original)
    def save_file(tensors, filename, *args, **kwargs):
        result = original(tensors, filename, *args, **kwargs)
        try:
            os.chmod(filename, mode)
        except OSError:  # not the owner, or the file is already gone
            pass

        return result

    safetensors.torch.save_file = save_file
    # transformers, peft, accelerate and our callbacks did `from safetensors.torch import save_file` at import
    # time, so they hold their own reference to the unwrapped function. Rebind those too.
    for module in list(sys.modules.values()):
        try:
            namespace = vars(module)
        except TypeError:  # extension modules without a __dict__
            continue

        for name, value in list(namespace.items()):
            if value is original:
                setattr(module, name, save_file)

    _installed = True
