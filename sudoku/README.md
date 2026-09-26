# relay / sudoku

Sudoku Extreme experiments from the *Learned Relay Representations* paper
(Table 1, paper §4.1). Code paths kept here are the ones that produce the
reported numbers: Mask-uniform CE, Rollout-buffer-only, Relay (BPTT), and
Relay-sg.

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd relay/sudoku
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

`xlm-core` is a submodule pinned to `ec2563c` (the SHA used for the paper
runs). It tracks `branch = main`. Do **not** run
`git submodule update --remote` on a first-time checkout — that floats past
the pinned SHA. Use it only when you intentionally want a newer harness.

### Python environment

Python 3.11. Create a virtual environment and install in this order:

```bash
python3.11 -m venv .venv_relay
source .venv_relay/bin/activate
# install a CUDA wheel first if `pip` would otherwise pull a CPU torch
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
cp .env.example .env
```

Edit `.env` and set `WANDB_ENTITY` to your entity, or pass
`loggers.wandb=null` on the Hydra command line to disable logging.

`.env` lives at `relay/sudoku/` — the `xlm` console script loads it from
the current working directory. `PROJECT_ROOT=.` is required so Hydra finds
`xlm_models.json` (`{"relay": "relay"}`).

## Data

Training and evaluation use
[`brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver`](https://huggingface.co/datasets/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver)
(train / test). First train downloads it into `HF_HOME` /
`HF_DATASETS_CACHE`. No separate `prepare_data` step is required.

To point at a different Hub repo or a local `datasets`-loadable directory:

```bash
export RELAY_SUDOKU_HF_DATASET=<your-hf-user>/sudoku-extreme-deduction
```

Hydra configs (`relay/configs/datasets/sudoku_extreme_*.yaml`) read this
variable via `oc.env` and fall back to the paper Hub id if it is unset.

Optional on-disk cache (not needed for Table 1):

```bash
export PROJECT_ROOT="$PWD"
xlm "job_type=prepare_data" "job_name=sudoku_extreme_prepare_data" "experiment=sudoku_extreme_mlm_uniform"
```

## Smoke test

A single-GPU sanity run to confirm the loop is wired up before launching the
full 300k training. All training goes through the `xlm` console script
(installed by `pip install -e xlm-core`; `python -m xlm` is equivalent):

```bash
export PROJECT_ROOT="$PWD"
xlm job_type=train \
  job_name=sudoku_extreme_smoke \
  experiment=sudoku_extreme_relay_bptt \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null
```

Substitute `experiment=sudoku_extreme_mlm_uniform` for the baseline.

## Reproducing Table 1

Shared knobs from the paper: 300k steps, val every 5k, batch 512, bf16-mixed,
lr `5e-4`, warmup 2000, inference threshold `0.15`. Relay-family jobs used
80GB-class GPUs at batch 512; drop `per_device_batch_size` and
`global_batch_size` together if VRAM is smaller. The paper averages seeds
`1`, `2`, `3` and both tying conditions (eight cells × three seeds).

`job_name` sets the log directory (`$LOG_DIR/<job_name>/`) and the W&B run
name, so keep it unique per run.

```bash
export PROJECT_ROOT="$PWD"
SEED=1   # paper averages seeds 1, 2, 3

# Mask-uniform CE
xlm job_type=train \
  job_name=sudoku_extreme_mlm_uniform_300k_untied_seed$SEED \
  experiment=sudoku_extreme_mlm_uniform \
  seed=$SEED \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=mlm_uniform \
  +tags.seed=$SEED

# Rollout-buffer-only
xlm job_type=train \
  job_name=sudoku_extreme_rollout_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  model=rotary_transformer_xtiny \
  loss.with_relay=false \
  loss.stop_grad_h_s=true \
  predictor.with_relay=false \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=rollout \
  +tags.seed=$SEED

# Relay-sg
xlm job_type=train \
  job_name=sudoku_extreme_relay_sg_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  loss.with_relay=true \
  loss.stop_grad_h_s=true \
  predictor.with_relay=true \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=relay_sg \
  +tags.seed=$SEED

# Relay (BPTT, T=2)
xlm job_type=train \
  job_name=sudoku_extreme_relay_bptt_steps2_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  loss.with_relay=true \
  loss.stop_grad_h_s=false \
  loss.num_steps=2 \
  predictor.with_relay=true \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=relay \
  +tags.seed=$SEED
```

For the tied vocab condition add `++model.tie_embeddings=true` and
`+tags.embed_tying=tied` (and swap `_untied` for `_tied` in `job_name`).
Pass `loggers.wandb=null` to skip W&B.

The Table 1 numbers (exact-match accuracy, token accuracy, mean NFE,
legal rate) are the validation-set metrics logged every
`val_check_interval` to W&B; see the metric-name list in
[`SUDOKU_COMMANDS.md`](SUDOKU_COMMANDS.md).

## Layout

| Path                 | Contents                                                                                  |
|----------------------|-------------------------------------------------------------------------------------------|
| `relay/`             | The Sudoku training package (model, loss, predictor, datamodule, metrics, sudoku tools). |
| `relay/configs/`     | Hydra configs (`experiment/`, `model/`, `model_type/`, `datamodule/`, `metrics/`, ...).   |
| `xlm-core/`          | Submodule: the XLM training harness used by every experiment.                             |
| `SUDOKU_COMMANDS.md` | Per-objective Hydra overrides and validation metric names.                                |

## W&B

Reported runs live under `https://wandb.ai/<your-wandb-entity>/BPTT-sudoku`.
The relevant filters are `+tags.sweep`, `+tags.objective`,
`+tags.embed_tying`, `+tags.seed`. Set `WANDB_ENTITY` in `.env`, or pass
`loggers.wandb=null` on the Hydra command line to disable logging entirely.
