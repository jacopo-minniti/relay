# TinyGSM relay ablation ladder

Three training objectives on TinyGSM question-to-Python, matching the Sudoku
rollout / RELAY (sg) / RELAY ladder. There is no mask-uniform MLM row here;
compare against xlm-core `experiment=tinygsm_mlm`.

| Objective | What it is | Hydra overrides |
|---|---|---|
| `rollout` | Rollout buffer, no relay carry | `loss.with_relay=false loss.stop_grad_h_s=true predictor.with_relay=false model=rotary_transformer_mlm` |
| `relay_sg` | Relay rollout, stop-gradient on `h` | `loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true` |
| `relay` | Relay rollout, truncated BPTT (`num_steps=2`) | `loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true` |

Shared knobs (the experiment already sets the relay row; the other rows are overrides):

```
experiment=tinygsm_relay_bptt
++trainer.precision=bf16-mixed
```

Reported metrics:

- `val/lm/accumulated_loss` — monitored training loss on the masked suffix.
- `val/prediction/code_exec_accuracy` — post-hoc `Gsm8kCodeEval` on the val holdout.
- `val/prediction/rollout_steps` — mean forward count (NFE) until the suffix is filled.
- `test/gsm8k_prediction/code_exec_accuracy` and `test/gsm8k_prediction/rollout_steps` — GSM8K test.

## Debug overfit (one committed row, no Hub download)

The row is `relay_tinygsm/debug_data/tinygsm_one/train.csv`. From `tinygsm/`:

```bash
source .venv_relay/bin/activate
export PROJECT_ROOT="$PWD" LOG_DIR=debug_logs HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# relay (BPTT, T=2), CPU
xlm job_type=train job_name=tinygsm_relay_overfit \
  experiment=tinygsm_relay_bptt debug=tinygsm_overfit trainer_strategy=cpu \
  model=rotary_transformer_relay_mlm_xtiny \
  loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true \
  trainer.max_steps=300 trainer.val_check_interval=50 \
  optimizer.lr=0.0003 lr_scheduler.num_warmup_steps=20

# relay-sg: same, with loss.stop_grad_h_s=true
# rollout: model=rotary_transformer_xtiny_mlm loss.with_relay=false \
#   loss.stop_grad_h_s=true predictor.with_relay=false
```

`debug=tinygsm_overfit` sets `overfit_batches=1`, batch size 1, and debug loggers
(no W&B). Attach a debugger with
`python -m debugpy --listen localhost:6768 --wait-for-client -m xlm` in place of `xlm`.

## Unity

Replica: `/scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm`.
`tinygsm/.env` sets
`DATA_DIR=/scratch4/workspace/dhruveshpate_umass_edu-learned-noise/learned-noise/data`
(full TinyGSM caches; the overfit job does not read it).

```bash
cd tinygsm
mkdir -p logs
RUNG=relay sbatch scripts/unity/debug_overfit.sbatch
RUNG=relay_sg sbatch scripts/unity/debug_overfit.sbatch
RUNG=rollout sbatch scripts/unity/debug_overfit.sbatch
```

Job ids are appended below after submission.

### Submitted jobs

All three finished `COMPLETED` on 2026-11-06. Logs are under
`tinygsm/logs/` in `/scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm`.
Final test metrics are `code_exec_accuracy=1.0` with rollout steps well under 256.

| Rung | Job | Partition | GPU | Log | Final loss | NFE |
|---|---|---|---|---|---|---|
| relay | 66850546 | superpod-a100 | A100-SXM4-80GB (`gpu020`) | `logs/tinygsm_relay_overfit_debug-66850546.out` | 0.028 | 8 |
| relay_sg | 66850856 | gpu-preempt-highvram | H100 80GB (`uri-gpu012`) | `logs/tinygsm_relay_overfit_debug-66850856.out` | 0.054 | 7 |
| rollout | 66850857 | superpod-a100 | A100-SXM4-80GB | `logs/tinygsm_relay_overfit_debug-66850857.out` | 0.021 | 5 |
