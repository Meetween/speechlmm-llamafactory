# Knowledge distillation (`train/kd`)

All KD code lives in this directory only. Upstream `llamafactory` files are **not** modified.

## Loss (decision doc)

`L = kd_ce_weight·CE + kd_jsd_weight·JSD(T) + kd_aut_mse_weight·MSE(AuT)` (defaults 0.5 / 0.5 / T=2 / 0.1).

## How to train

1. Set `PYTHONPATH` to include `speechlmm-v2/src` and `llamafactory/src` (see `Knowledge Distillation/kd/scripts/slurm/kd_common_env.sh`).
2. Use the KD CLI (registers hooks via `bootstrap.py`):

```bash
cd speechlmm-v2/llamafactory
export FORCE_TORCHRUN=1   # multi-GPU / DeepSpeed
python -m llamafactory.train.kd.launcher train /path/to/kd/configs/kd_stage0_mmsu.yaml
```

Single process:

```bash
python -m llamafactory.train.kd train /path/to/config.yaml
```

Do **not** use `llamafactory-cli train` with `stage: kd` until Meetween merges `FinetuningArgumentsKD` into `hparams/finetuning_args.py` and `tuner.py`.

## Modules

| File | Role |
|------|------|
| `bootstrap.py` | Runtime registration (parser, tuner dispatch, SpeechLMM config load) |
| `hparams.py` | `KDArguments`, `FinetuningArgumentsKD` |
| `workflow.py` | Dataset, teacher `ref_model`, trainer setup |
| `trainer.py` | Dual forward + composite loss |
| `losses.py` | JSD, CE, AuT MSE |
| `freeze.py` | Stage 0 projector-only |
| `cli.py` / `launcher.py` | Entry points |

## Promoting upstream

When ready, move `KDArguments` into `hparams/finetuning_args.py`, add `elif stage == "kd"` in `train/tuner.py`, and drop runtime patches from `bootstrap.py`.
