#!/bin/bash
# One-GPU TinyGSM debug overfit (xtiny, 300 steps, one committed CSV row).
# Submit from tinygsm/:
#   mkdir -p logs
#   RUNG=relay sbatch scripts/unity/debug_overfit.sbatch
#   RUNG=relay_sg sbatch scripts/unity/debug_overfit.sbatch
#   RUNG=rollout sbatch scripts/unity/debug_overfit.sbatch
#SBATCH --job-name            tinygsm_relay_overfit_debug
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --gres                gpu:1
#SBATCH --cpus-per-task       5
#SBATCH --mem                 20GB
#SBATCH --time                01:00:00
#SBATCH --partition           superpod-a100,gpu-highvram,gpu-preempt-highvram,gpu,gpu-preempt
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

RUNG="${RUNG:-relay}"
case "$RUNG" in
  relay)
    extra=(
      model=rotary_transformer_relay_mlm_xtiny
      loss.with_relay=true
      loss.stop_grad_h_s=false
      loss.num_steps=2
      predictor.with_relay=true
    )
    ;;
  relay_sg)
    extra=(
      model=rotary_transformer_relay_mlm_xtiny
      loss.with_relay=true
      loss.stop_grad_h_s=true
      predictor.with_relay=true
    )
    ;;
  rollout)
    extra=(
      model=rotary_transformer_xtiny_mlm
      loss.with_relay=false
      loss.stop_grad_h_s=true
      predictor.with_relay=false
    )
    ;;
  *)
    echo "Unknown RUNG=$RUNG (expected relay, relay_sg, or rollout)" >&2
    exit 1
    ;;
esac

xlm job_type=train \
  job_name="tinygsm_${RUNG}_overfit" \
  experiment=tinygsm_relay_bptt \
  debug=tinygsm_overfit \
  trainer.max_steps=300 \
  trainer.val_check_interval=50 \
  optimizer.lr=0.0003 \
  lr_scheduler.num_warmup_steps=20 \
  trainer.precision=bf16-mixed \
  "${extra[@]}"
