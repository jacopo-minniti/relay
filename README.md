# RELAY: Learned Relay Representations for Forward-Thinking Discrete Diffusion Models

Code release for the paper *Learned Relay Representations for Forward-Thinking Discrete Diffusion Models* (NeurIPS 2026 submission). When Masked Diffusion Models (MDMs) generate sequences through iterative refinement, the rich internal computation accumulated over masked positions is discarded at the end of each forward pass — forcing every subsequent denoising step to start from scratch. We call this the **hard reset** problem. To address it, we propose **RELAY**: at each denoising step the model carries its last-layer hidden states forward as a learned relay, giving the next forward pass direct access to prior continuous computation. The relay is trained end-to-end via truncated backpropagation through time (BPTT), shaping it to be maximally informative for the next several denoising steps. RELAY is architecture-agnostic, leaves the inference-time decoding procedure of MDMs unchanged, and is compatible with block diffusion and KV caching.

We validate the design choices on a Sudoku-based planning task, then scale RELAY to **Fast-dLLM v2 1.5B** (a state-of-the-art block-diffusion language model) — outperforming standard supervised fine-tuning on coding tasks while decreasing inference latency by up to 32%.

> **Anonymized public mirror.** A read-only anonymized copy of this repository is available at <https://anonymous.4open.science/r/relay-1D24>. The mirror is **not** auto-synced from GitHub; after pushing changes here, click *Force update* on the anon4open page (see [Anonymized release](#anonymized-release) below).

---

## Repository structure

```
relay/
├── sudoku/         design study on Sudoku-Extreme  (Table 1 of the paper)
└── fast-dllm-v2/   adaptation of Fast-dLLM v2 1.5B (Table 2 of the paper)
```

The two subdirectories are intentionally independent: each ships its own Python environment, dataset preparation pipeline, training launcher, and evaluation pipeline. Pick the subdirectory that matches the experiments you want to reproduce.

| Sub-project | Paper section | Environment |
|---|---|---|
| [`sudoku/`](sudoku) | §4.1 (Table 1, Figure 2) | `.venv_relay` venv (`python -m venv`, Python 3.11.10) |
| [`fast-dllm-v2/`](fast-dllm-v2) | §4.2 (Table 2, Figure 3) | `relay` conda env (Python 3.10) |

> Both sub-projects log to Weights & Biases by default. Set `WANDB_ENTITY` to your own entity in each sub-project's `.env` (or pass `loggers.wandb=null` to the Sudoku launcher) to log to your account or disable logging.

---

## Reproducing the paper

### Table 1 — Sudoku-Extreme (`sudoku/`)

Each row is one training objective × one weight-tying condition. The four objectives map to the per-objective Hydra overrides documented in [`sudoku/SUDOKU_COMMANDS.md`](sudoku/SUDOKU_COMMANDS.md):

| Objective | What it is | Key Hydra overrides |
|---|---|---|
| **MLM** | Mask-uniform CE baseline | `experiment=sudoku_extreme_mlm_uniform` |
| **Rollout** | Rollout-buffer-only (`K=2`, no relay) | `experiment=sudoku_extreme_relay_bptt loss.with_relay=false predictor.with_relay=false loss.stop_grad_h_s=true` |
| **RELAY (sg)** | Relay rollout with stop-gradient on `h` | `experiment=sudoku_extreme_relay_bptt loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true` |
| **RELAY** | Full method: `K=2` BPTT through the relay (paper Algorithm 1) | `experiment=sudoku_extreme_relay_bptt loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true` |

Each of the four objectives is run in two weight-tying conditions (`tied` and `untied`), giving the eight rows of Table 1; the paper averages over three training seeds.

```bash
git clone --recurse-submodules https://github.com/jacopo-minniti/relay.git
cd relay/sudoku
# if you already cloned without submodules:
#   git submodule update --init --recursive

# 1. Environment + .env setup (one-time). See sudoku/README.md for details.
python3.11 -m venv .venv_relay
source .venv_relay/bin/activate
# install a CUDA wheel first if `pip` would otherwise pull a CPU torch
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
cp .env.example .env   # set WANDB_ENTITY (or pass loggers.wandb=null)

# 2. Quick smoke test (single GPU, 200 steps)
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

# 3. Reproduce Table 1 — one `xlm job_type=train` command per objective.
#    Full commands (Mask-uniform / Rollout / Relay-sg / Relay, plus tied
#    and seeds 1/2/3) are in sudoku/README.md.
```

The first train downloads [`brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver`](https://huggingface.co/datasets/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver) (no separate `prepare_data` step). The Table 1 numbers (exact-match accuracy, token accuracy, mean NFE, legal rate) are the validation-set metrics logged each `val_check_interval` to W&B; see [`sudoku/README.md`](sudoku/README.md) and [`sudoku/SUDOKU_COMMANDS.md`](sudoku/SUDOKU_COMMANDS.md).

### Table 2 — Fast-dLLM v2 1.5B (`fast-dllm-v2/`)

Each row is a 200-step adaptation of the off-the-shelf [`Efficient-Large-Model/Fast_dLLM_v2_1.5B`](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B) on the OpenCodeInstruct + OpenMathInstruct-2 c40m60 mixture (24 000 code rows + 36 000 math rows = 60 000 rows; effective batch size 64; learning rate 5e-6). EvalPlus pass@1 is reported at `threshold=0.85`, BD block 32, sub-block 8, exactly as in Wu et al. (2025b).

| Row | What it is | Launch flags |
|---|---|---|
| **Fast-dLLM-v2 (1.5B)** | Off-the-shelf base model | (no training) |
| **Vanilla SFT** | `--loss_type mlm` (no relay) | `train_scripts/finetune_opencode_openmath.sbatch` |
| **Rollout** | `--loss_type bptt --bptt_use_relay 0` (on-policy 2-step rollout, relay channel disabled) | `USE_RELAY=0 sbatch train_scripts/finetune_opencode_openmath_bptt.sbatch` |
| **RELAY (sg)** | `--loss_type bptt --bptt_use_relay 1 --bptt_stop_grad_h_s 1` | `USE_RELAY=1 BPTT_STOP_GRAD_H_S=1 sbatch train_scripts/finetune_opencode_openmath_bptt.sbatch` |
| **RELAY** | `--loss_type bptt --bptt_use_relay 1 --bptt_stop_grad_h_s 0` | `USE_RELAY=1 BPTT_STOP_GRAD_H_S=0 sbatch train_scripts/finetune_opencode_openmath_bptt.sbatch` |

```bash
# From this repository root (the directory that contains ``fast-dllm-v2/`` and this README).
cd fast-dllm-v2/v2

# 1. Environment (one-time). Defaults assume CONDA_ROOT=${HOME}/miniconda3
#    and a conda env named "relay"; both are overridable via env vars.
conda create -n relay python=3.10 pip ipykernel -y
conda activate relay
pip install -e '.[eval]'   # core LMFlow + pinned EvalPlus for Table 2 reproduce
# Train-only (lighter deps): pip install -e .

# 2. Build the c40m60 mixture (one-time, ~5 minutes; reads from the HF Hub).
python scripts/prep_opencode_openmath_mix.py \
  --out_dir data/opencode_openmath_60k_c40m60 \
  --code_rows 24000 --math_rows 36000 --require_code_def

# 3. Quick local smoke test (no SLURM, 1 GPU, 5 optimizer steps).
DRY_RUN=1 BLOCK_SIZE=512 NUM_TRAIN_EPOCHS=1 SAVE_STEPS=5 \
  bash scripts/launch_opencode_openmath_c40m60_4run.sh

# 4. Reproduce Table 2 on a SLURM cluster (4 jobs, 2x A100-80GB each).
bash scripts/launch_opencode_openmath_c40m60_4run.sh

# 5. Evaluate any saved checkpoint with EvalPlus (HumanEval+ / MBPP+).
#    --model_path accepts either a HF Hub id or a local Trainer checkpoint dir.
python scripts/generate_evalplus_jsonl.py \
  --model_path output_models/opencode_openmath_60k_c40m60_1p5B/<run-dir>/checkpoint-200 \
  --dataset humaneval --threshold 0.85 --use_carry           # use --use_carry only for relay
evalplus.evaluate --dataset humaneval --samples <jsonl-out>  # pinned to evalplus==0.3.1
```

Cluster-specific knobs (`CONDA_ROOT`, `CONDA_ENV`, `RESERVATION`, partition, wall-clock) are picked up from environment variables; copy [`fast-dllm-v2/v2/.env.example`](fast-dllm-v2/v2/.env.example) to `.env` to set defaults. The kept sbatch wrappers (`train_scripts/*.sbatch`) all use `--mail-user=$USER` and `${HOME}/miniconda3` defaults so they require no edits to run on a different cluster.

#### Released RELAY checkpoints

The two adapted models that produce the **RELAY** and **RELAY (sg)** rows of Table 2 are published on the Hugging Face Hub:

| Row | Hub repo |
|---|---|
| **RELAY** | [`brozonoyer/relay-fastdllm-v2-c40m60-relay-step200`](https://huggingface.co/brozonoyer/relay-fastdllm-v2-c40m60-relay-step200) |
| **RELAY (sg)** | [`brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200`](https://huggingface.co/brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200) |

Both ship with `use_relay=True` / `relay_layer=-1` in `config.json` and a `model.relay_layer_norm.{weight,bias}` tensor in the safetensors shard, plus a self-contained `configuration.py` / `modeling.py` (`auto_map`-wired), so `trust_remote_code=True` is enough to load and run them — no checkout of this repo required for inference. The vendored `fast-dllm-v2/v2/src/lmflow/models/fast_dllm/{configuration,modeling}.py` here is identical to what each repo bundles, so you can also load them against the in-tree source.

Table 2 EvalPlus numbers in the paper were produced with the same defaults as
``Fast-dLLM/v2/scripts/submit_eval.py`` (``accelerate launch eval.py``): **`--max_new_tokens 2048`** and **`--batch_size 32`** (threshold **0.85**, BD block **32**, sub-block **8**). The standalone generator defaults to **`max_new_tokens=512`** for quicker smoke runs—that **lowers average NFE** because the masked-diffusion loop cannot allocate as many continuation blocks; see the comparison table below.

Reproduce **RELAY** / **RELAY (sg)** from the Hub with **paper-aligned** decoding (add ``--nfe_stats_json`` for sidecar NFE JSON; absolute counts only):

```bash
# From this repository root (the directory that contains ``fast-dllm-v2/`` and this README).
cd fast-dllm-v2/v2
conda activate relay    # env where you ran ``pip install -e '.[eval]'``
mkdir -p evalplus_results

MNT=2048
BS=32

# --- HumanEval+ --------------------------------------------------------------
python scripts/generate_evalplus_jsonl.py \
  --model_path brozonoyer/relay-fastdllm-v2-c40m60-relay-step200 \
  --dataset humaneval --use_carry --threshold 0.85 \
  --max_new_tokens "${MNT}" --batch_size "${BS}" \
  --output_jsonl evalplus_results/relay_humaneval_mnt"${MNT}"_bs"${BS}".jsonl \
  --nfe_stats_json evalplus_results/relay_humaneval_mnt"${MNT}"_bs"${BS}"_nfe.json \
  && evalplus.evaluate --dataset humaneval --samples evalplus_results/relay_humaneval_mnt"${MNT}"_bs"${BS}".jsonl

python scripts/generate_evalplus_jsonl.py \
  --model_path brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200 \
  --dataset humaneval --use_carry --threshold 0.85 \
  --max_new_tokens "${MNT}" --batch_size "${BS}" \
  --output_jsonl evalplus_results/relay_sg_humaneval_mnt"${MNT}"_bs"${BS}".jsonl \
  --nfe_stats_json evalplus_results/relay_sg_humaneval_mnt"${MNT}"_bs"${BS}"_nfe.json \
  && evalplus.evaluate --dataset humaneval --samples evalplus_results/relay_sg_humaneval_mnt"${MNT}"_bs"${BS}".jsonl

# --- MBPP+ -------------------------------------------------------------------
python scripts/generate_evalplus_jsonl.py \
  --model_path brozonoyer/relay-fastdllm-v2-c40m60-relay-step200 \
  --dataset mbpp --use_carry --threshold 0.85 \
  --max_new_tokens "${MNT}" --batch_size "${BS}" \
  --output_jsonl evalplus_results/relay_mbpp_mnt"${MNT}"_bs"${BS}".jsonl \
  --nfe_stats_json evalplus_results/relay_mbpp_mnt"${MNT}"_bs"${BS}"_nfe.json \
  && evalplus.evaluate --dataset mbpp --samples evalplus_results/relay_mbpp_mnt"${MNT}"_bs"${BS}".jsonl

python scripts/generate_evalplus_jsonl.py \
  --model_path brozonoyer/relay-fastdllm-v2-c40m60-relay-sg-step200 \
  --dataset mbpp --use_carry --threshold 0.85 \
  --max_new_tokens "${MNT}" --batch_size "${BS}" \
  --output_jsonl evalplus_results/relay_sg_mbpp_mnt"${MNT}"_bs"${BS}".jsonl \
  --nfe_stats_json evalplus_results/relay_sg_mbpp_mnt"${MNT}"_bs"${BS}"_nfe.json \
  && evalplus.evaluate --dataset mbpp --samples evalplus_results/relay_sg_mbpp_mnt"${MNT}"_bs"${BS}".jsonl
```

Compare ``avg_nfe`` in each ``*_nfe.json`` with Table 2. Omit ``--nfe_stats_json`` if you only need EvalPlus scores. Use ``--use_carry`` for these relay Hub checkpoints.

##### Table 2 vs local ``max_new_tokens=512`` rerun (same Hub weights)

Pass@1 is from ``evalplus.evaluate`` on the saved JSONL; NFE is ``avg_nfe`` from the sidecar JSON (generator **defaults**: ``max_new_tokens=512``, ``batch_size=4``). Paper column is the printed Table 2 (2048-token eval infra).

| Method | | HumanEval Base↑ | HumanEval Plus↑ | HumanEval NFE↓ | MBPP Base↑ | MBPP Plus↑ | MBPP NFE↓ |
|--------|---|-----------------|-----------------|----------------|------------|------------|-----------|
| **Paper Table 2** | RELAY (sg) | 38.4% | 35.4% | 104.4 | 43.1% | 39.2% | 80.1 |
| | **RELAY** | **42.1%** | **37.2%** | **88.3** | **46.6%** | **41.5%** | **78.8** |
| **512 rerun** (this repo, ``evalplus_results/``) | RELAY (sg) | 41.5% | 37.8% | 82.3 | 42.6% | 38.4% | 69.6 |
| | RELAY | 42.1% | 37.2% | 79.3 | 46.8% | 41.8% | 66.5 |

Rerun with ``MNT=2048``, ``BS=32`` above to match paper NFE accounting; accuracies may still differ slightly from Table 2 due to EvalPlus / tokenizer revisions or GPU nondeterminism, but NFE should align with archived ``eval_results_nfe_v2`` runs.

---

## Anonymized release

The paper points readers at the anonymized mirror <https://anonymous.4open.science/r/relay-1D24>, which is a **manual snapshot** of this GitHub repository. Pushing to GitHub does **not** automatically refresh the mirror; instead:

1. Push the changes to the GitHub repository.
2. Open the anon4open URL above and click the **Force update** button (top-right). The service re-fetches the public state of the repo and rebuilds the snapshot.
3. Optionally verify by querying the snapshot API: `curl https://anonymous.4open.science/api/repo/relay-1D24` should report the new commit timestamp.

For the camera-ready, request a fresh anonymous slug if the snapshot has accidentally been updated past the submission deadline.

---

## Citation

```bibtex
@inproceedings{relay2026,
  title  = {Learned Relay Representations for Forward-Thinking Discrete Diffusion Models},
  author = {Anonymous},
  booktitle = {Submitted to NeurIPS 2026},
  year   = {2026}
}
```

The Fast-dLLM v2 sub-project additionally uses code from Wu et al. ([arXiv:2509.26328](https://arxiv.org/abs/2509.26328)); its citation is in [`fast-dllm-v2/v2/README.md`](fast-dllm-v2/v2/README.md).
