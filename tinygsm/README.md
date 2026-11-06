# TinyGSM relay ablation

Rollout, RELAY (sg), and RELAY (truncated BPTT) on TinyGSM question-to-Python.
The data pipeline, seq2seq collators, and `Gsm8kCodeEval` come from xlm-core
and the `mlm` package. This directory adds the relay model, the rollout-buffer
loss, and a confidence predictor that carries hidden states across denoising
steps.

| Objective | Overrides |
|---|---|
| Rollout | `loss.with_relay=false predictor.with_relay=false loss.stop_grad_h_s=true model=rotary_transformer_mlm` |
| RELAY (sg) | `loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true` |
| RELAY | `loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true` |

Full commands: [TINYGSM_COMMANDS.md](TINYGSM_COMMANDS.md).

## Setup

```bash
cd tinygsm
git submodule update --init xlm-core
python3.11 -m venv .venv_relay
source .venv_relay/bin/activate
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
cp .env.example .env   # set WANDB_ENTITY, or pass loggers.wandb=null
export PROJECT_ROOT="$PWD"
```

On Unity, use `scripts/unity/setup_venv.sh` (conda prefix env) instead of the venv above.
Point `DATA_DIR` in `.env` at the prepared TinyGSM cache. The one-row debug overfit
does not use `DATA_DIR`; the CSV is in `relay_tinygsm/debug_data/`.

## Train

Prepare the TinyGSM manual cache once (rank 0), then train:

```bash
xlm job_type=prepare_data experiment=tinygsm_relay_bptt num_dataset_workers=8

xlm job_type=train job_name=tinygsm_relay experiment=tinygsm_relay_bptt
```

Pass `loggers.wandb=null` to skip W&B. Validation logs
`val/prediction/code_exec_accuracy` (generated Python executed against the gold number)
and `val/prediction/rollout_steps` (mean NFE).

## Debug overfit on one row

```bash
export PROJECT_ROOT="$PWD" HF_HUB_OFFLINE=1
xlm job_type=train job_name=tinygsm_relay_overfit \
  experiment=tinygsm_relay_bptt debug=tinygsm_overfit trainer_strategy=cpu \
  model=rotary_transformer_relay_mlm_xtiny \
  trainer.max_steps=300 trainer.val_check_interval=50 \
  optimizer.lr=0.0003 lr_scheduler.num_warmup_steps=20 \
  loggers.wandb=null
```

`debug=tinygsm_overfit` trains on `relay_tinygsm/debug_data/tinygsm_one/train.csv` only.
