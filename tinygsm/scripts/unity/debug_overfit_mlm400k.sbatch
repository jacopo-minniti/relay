#!/bin/bash
# Full-size TinyGSM debug overfit, warm-started from the 400k MLM model-only file.
# Submit from tinygsm/ after that file exists:
#   mkdir -p logs
#   RUNG=relay sbatch scripts/unity/debug_overfit_mlm400k.sbatch
#   RUNG=relay_sg sbatch scripts/unity/debug_overfit_mlm400k.sbatch
#   RUNG=rollout sbatch scripts/unity/debug_overfit_mlm400k.sbatch
#   RUNG=mlm sbatch --job-name=tinygsm_mlm_overfit_mlm400k scripts/unity/debug_overfit_mlm400k.sbatch
#SBATCH --job-name            tinygsm_relay_overfit_mlm400k
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --gres                gpu:1
#SBATCH --cpus-per-task       5
#SBATCH --mem                 20GB
#SBATCH --time                01:00:00
#SBATCH --partition           superpod-a100
#SBATCH --constraint          vram80,bf16
#SBATCH --open-mode           append
#SBATCH --output              logs/%x-%j.out
#SBATCH --mail-type           BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user           dhruveshpate@umass.edu

set -euo pipefail
cd /scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm/tinygsm

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.venv_relay

nvidia-smi -L
export PROJECT_ROOT="$PWD"
export XLM_MODELS_PATH="$PWD/xlm-core/xlm-models"

RUNG="${RUNG:-relay}"
experiment=tinygsm_relay_bptt
case "$RUNG" in
  relay)
    extra=(
      model=rotary_transformer_relay_mlm
      loss.with_relay=true
      loss.stop_grad_h_s=false
      loss.num_steps=2
      predictor.with_relay=true
      strict_model_only_load=false
    )
    ;;
  relay_sg)
    extra=(
      model=rotary_transformer_relay_mlm
      loss.with_relay=true
      loss.stop_grad_h_s=true
      predictor.with_relay=true
      strict_model_only_load=false
    )
    ;;
  rollout)
    extra=(
      model=rotary_transformer_mlm
      loss.with_relay=false
      loss.stop_grad_h_s=true
      predictor.with_relay=false
      strict_model_only_load=true
    )
    ;;
  mlm)
    experiment=tinygsm_mlm
    extra=(
      model=rotary_transformer_mlm
      strict_model_only_load=true
      '~predictor.log_rollout_diagnostics'
    )
    ;;
  *)
    echo "Unknown RUNG=$RUNG (expected relay, relay_sg, rollout, or mlm)" >&2
    exit 1
    ;;
esac

MODEL_ONLY=/scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm/checkpoints/tinygsm_mlm_v2/0-400000.pth

xlm job_type=train \
  job_name="tinygsm_${RUNG}_overfit_mlm400k" \
  experiment="$experiment" \
  debug=tinygsm_overfit \
  trainer.max_steps=300 \
  trainer.val_check_interval=50 \
  optimizer.lr=0.0003 \
  lr_scheduler.num_warmup_steps=20 \
  trainer.precision=bf16-mixed \
  +model_only_checkpoint_path="$MODEL_ONLY" \
  "${extra[@]}"
