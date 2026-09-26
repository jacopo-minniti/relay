#!/usr/bin/env bash
# Submit the four reported OpenCode/OpenMath c40m60 runs concurrently.
# These reproduce the adapted rows of Table 2 (paper) for the c40m60 mixture:
#   - vanilla   : Vanilla SFT (loss_type=mlm, no relay)
#   - rollout   : Rollout     (loss_type=bptt, use_relay=False; on-policy rollout, relay channel disabled)
#   - relay_sg  : RELAY (sg)  (loss_type=bptt, use_relay=True, stop_grad_h_s=1)
#   - relay     : RELAY       (loss_type=bptt, use_relay=True, stop_grad_h_s=0)
#
# Run from Fast-dLLM/v2 after data prep:
#   bash scripts/launch_opencode_openmath_c40m60_4run.sh
#
# Set DRY_RUN=1 to print commands without submitting.
# Set RESERVATION=<name> to add --reservation <name> to every sbatch call.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

recipe_tag=${RECIPE_TAG:-opencode_openmath_60k_c40m60}
dataset_path=${DATASET_PATH:-data/${recipe_tag}/train_conversation}
if [ ! -d "${dataset_path}" ]; then
    echo "ERROR: dataset path does not exist: ${dataset_path}" >&2
    echo "Run:" >&2
    echo "  python scripts/prep_opencode_openmath_mix.py \\" >&2
    echo "    --out_dir data/${recipe_tag} \\" >&2
    echo "    --code_rows 24000 --math_rows 36000 --require_code_def" >&2
    exit 1
fi

reservation_arg=()
if [ -n "${RESERVATION:-}" ]; then
    reservation_arg=(--reservation "${RESERVATION}")
fi

common_export="ALL,DATASET_PATH=${dataset_path},RECIPE_TAG=${recipe_tag},BLOCK_SIZE=${BLOCK_SIZE:-2048},DISABLE_GROUP_TEXTS=${DISABLE_GROUP_TEXTS:-1},NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3},LEARNING_RATE=${LEARNING_RATE:-5e-6},SAVE_STEPS=${SAVE_STEPS:-200},SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0},PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}"

submit() {
    local label=$1
    shift
    local cmd=(sbatch "${reservation_arg[@]}" "$@")
    echo "[$label] ${cmd[*]}" >&2
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    local out
    out=$("${cmd[@]}")
    echo "[$label] ${out}" >&2
    echo "${out}" | awk '{print $NF}'
}

declare -A job_ids=()

job_ids[vanilla]=$(submit vanilla \
    --job-name=ft_c40m60_vanilla_1p5B \
    --export="${common_export}" \
    train_scripts/finetune_opencode_openmath.sbatch)

job_ids[rollout]=$(submit rollout \
    --job-name=ft_c40m60_rollout_1p5B \
    --export="${common_export},USE_RELAY=0,BPTT_STOP_GRAD_H_S=0" \
    train_scripts/finetune_opencode_openmath_bptt.sbatch)

job_ids[relay_sg]=$(submit relay_sg \
    --job-name=ft_c40m60_relay_sg_1p5B \
    --export="${common_export},USE_RELAY=1,BPTT_STOP_GRAD_H_S=1" \
    train_scripts/finetune_opencode_openmath_bptt.sbatch)

job_ids[relay]=$(submit relay \
    --job-name=ft_c40m60_relay_1p5B \
    --export="${common_export},USE_RELAY=1,BPTT_STOP_GRAD_H_S=0" \
    train_scripts/finetune_opencode_openmath_bptt.sbatch)

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

printf "Submitted jobs:\n"
for key in vanilla rollout relay_sg relay; do
    printf "  %-20s %s\n" "${key}" "${job_ids[$key]}"
done

echo
echo "Eval is NOT auto-submitted. After checkpoints are written, launch the"
echo "eval sweep manually per run, e.g.:"
echo "  MODEL_ROOT=output_models/${recipe_tag}_1p5B/vanilla_nopack2048_bs2x16x2_ep3_lr5e-6 \\"
echo "  USE_CARRY=0 CHECKPOINTS=200,400,600,800,final \\"
echo "  sbatch train_scripts/submit_opencode_openmath_eval_sweep.sbatch"
