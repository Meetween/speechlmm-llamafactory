from types import SimpleNamespace

import pytest
import torch

from llamafactory.extras.constants import IGNORE_INDEX
from speechlmm.models.talker_loss import pad_codec_batch


@pytest.mark.runs_on(["cpu", "mps"])
def test_single_codebook_padding():
    labels, mask = pad_codec_batch([[1, 2, 3], [4]], num_codebooks=1, ignore_index=IGNORE_INDEX)
    assert labels.tolist() == [[1, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]
    assert mask.tolist() == [True, True]


@pytest.mark.runs_on(["cpu", "mps"])
def test_multi_codebook_padding():
    labels, mask = pad_codec_batch([[[1, 2], [3, 4]], None], num_codebooks=2, ignore_index=IGNORE_INDEX)
    assert tuple(labels.shape) == (2, 2, 2)
    assert labels[1].tolist() == [[IGNORE_INDEX, IGNORE_INDEX], [IGNORE_INDEX, IGNORE_INDEX]]
    assert mask.tolist() == [True, False]


@pytest.mark.runs_on(["cpu", "mps"])
def test_codebook_count_mismatch_raises():
    with pytest.raises(ValueError, match="talker_num_codebooks"):
        pad_codec_batch([[[1, 2]]], num_codebooks=16)


@pytest.mark.runs_on(["cpu", "mps"])
def test_dummy_codec_when_batch_has_none():
    labels, mask = pad_codec_batch([None, None], num_codebooks=16, dummy=True)
    assert tuple(labels.shape) == (2, 1, 16)
    assert not bool(mask.any())
    assert torch.equal(labels, torch.zeros_like(labels))
