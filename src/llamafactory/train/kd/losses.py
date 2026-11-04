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

import math

import torch
import torch.nn.functional as F


def jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    ignore_index: int,
) -> torch.Tensor:
    r"""Mean Jensen-Shannon divergence over the next-token positions that have a label."""
    mask = labels[:, 1:] != ignore_index
    if not mask.any():
        return student_logits.sum() * 0.0

    # Index before the fp32 cast so only labeled positions are materialized at full vocab size.
    log_p_s = F.log_softmax(student_logits[:, :-1][mask].float() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher_logits[:, :-1][mask].float() / temperature, dim=-1)
    log_m = torch.logaddexp(log_p_s, log_p_t) - math.log(2.0)
    kl_s = (log_p_s.exp() * (log_p_s - log_m)).sum(dim=-1)
    kl_t = (log_p_t.exp() * (log_p_t - log_m)).sum(dim=-1)
    return (0.5 * (kl_s + kl_t)).mean()
