# Copyright 2026 the LlamaFactory / SpeechLMM team.

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Seq2SeqTrainer

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class DynamicTrainingLoggingTests(unittest.TestCase):
    def test_dynamic_metrics_preserve_the_base_training_loss_log(self) -> None:
        trainer = object.__new__(CustomSeq2SeqTrainer)
        trainer.dynamic_batch_plan = object()
        trainer._last_dynamic_metrics_step = 0
        trainer.state = SimpleNamespace(global_step=1)
        trainer.control = SimpleNamespace(should_log=True)
        trainer._dynamic_step_metrics = lambda current_step: {"dynamic_global_step": float(current_step)}

        def fake_log(metrics, start_time=None):
            assert metrics == {"dynamic_global_step": 1.0}
            # This mirrors CallbackHandler.on_log in Transformers.
            trainer.control.should_log = False

        trainer.log = fake_log
        base_should_log: list[bool] = []

        def fake_base_maybe_log(instance, *args, **kwargs):
            base_should_log.append(instance.control.should_log)

        with patch.object(
            Seq2SeqTrainer,
            "_maybe_log_save_evaluate",
            new=fake_base_maybe_log,
        ):
            trainer._maybe_log_save_evaluate(
                tr_loss=torch.tensor(1.0),
                grad_norm=None,
                model=None,
                trial=None,
                epoch=0,
                ignore_keys_for_eval=None,
                start_time=0.0,
            )

        assert base_should_log == [True]


if __name__ == "__main__":
    unittest.main()
