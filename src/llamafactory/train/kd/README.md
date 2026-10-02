# Knowledge distillation (`stage: kd`)

Launch with the normal CLI. `llamafactory-cli train` dispatches `stage: kd`.

```yaml
stage: kd
finetuning_type: full
ref_model: /path/to/frozen/teacher
kd_ce_weight: 0.5
kd_jsd_weight: 0.5
kd_temperature: 1.0
kd_enable_aut_mse: false
```

The loss is `kd_ce_weight * CE + kd_jsd_weight * JSD(T)` on assistant tokens.
`kd_enable_aut_mse: true` adds an audio-encoder MSE term. It is off by default.

| File | Role |
|------|------|
| `workflow.py` | SFT dataset and collator, frozen teacher via `ref_model` |
| `trainer.py` | Student forward, teacher forward, composite loss |
| `losses.py` | JSD, CE, optional audio-encoder MSE |
| `freeze.py` | Optional projector-only freeze (`kd_stage0_projector_only`) |
