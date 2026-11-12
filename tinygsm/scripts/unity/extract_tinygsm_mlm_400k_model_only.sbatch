#!/bin/bash
# Extract EMA model weights from the tinygsm_mlm_v2 400k Lightning checkpoint.
# Submit from tinygsm/:
#   mkdir -p logs
#   sbatch scripts/unity/extract_tinygsm_mlm_400k_model_only.sbatch
#SBATCH --job-name            tinygsm_mlm_extract_400k
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --gres                gpu:1
#SBATCH --cpus-per-task       5
#SBATCH --mem                 64G
#SBATCH --time                02:00:00
#SBATCH --partition           superpod-a100
#SBATCH --constraint          vram80,bf16
#SBATCH --open-mode           append
#SBATCH --output              logs/%x-%j.out
#SBATCH --mail-type           BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user           dhruveshpate@umass.edu
#SBATCH --requeue

set -euo pipefail
cd /scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm/tinygsm

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.venv_relay

export PROJECT_ROOT="$PWD"
export XLM_MODELS_PATH="$PWD/xlm-core/xlm-models"

CKPT=/project/pi_mccallum_umass_edu/dhruveshpate_umass_edu/xlm-models/tinygsm_mlm_v2/checkpoints/0-400000.ckpt
OUT=/scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm/checkpoints/tinygsm_mlm_v2/0-400000.pth

xlm job_type=extract_checkpoint \
  job_name=tinygsm_mlm_v2_extract_400k \
  experiment=tinygsm_mlm \
  +post_training=default \
  post_training.checkpoint_path="$CKPT" \
  post_training.apply_ema=true \
  post_training.model_state_dict_path="$OUT" \
  post_training.repo_id=null

python - <<'PY'
import torch
from relay_tinygsm.model import RotaryTransformerRelayMLMModel

path = "/scratch3/workspace/dhruveshpate_umass_edu-relay/relay-tinygsm/checkpoints/tinygsm_mlm_v2/0-400000.pth"
sd = torch.load(path, map_location="cpu", weights_only=True)
print("n_tensors", len(sd))
if len(sd) == 0:
    raise SystemExit("extracted state dict is empty")
vocab, d_model = sd["embed_tokens.weight"].shape
layer_ids = {
    int(key.split(".")[1])
    for key in sd
    if key.startswith("encoder.") and key.split(".")[1].isdigit()
}
n_layers = max(layer_ids) + 1
relay = RotaryTransformerRelayMLMModel(
    num_embeddings=vocab,
    d_model=d_model,
    num_layers=n_layers,
    nhead=12,
    dim_feedforward=4 * d_model,
    rotary_emb_dim=64,
    max_length=516,
)
own = relay.state_dict()
missing = sorted(set(own) - set(sd))
unexpected = sorted(set(sd) - set(own))
print("missing", missing)
print("unexpected", unexpected)
if missing != ["relay_layer_norm.bias", "relay_layer_norm.weight"]:
    raise SystemExit(f"unexpected missing keys: {missing}")
if unexpected:
    raise SystemExit(f"unexpected extra keys: {unexpected}")
for key, value in sd.items():
    if value.shape != own[key].shape:
        raise SystemExit(f"shape mismatch {key}: {tuple(value.shape)} vs {tuple(own[key].shape)}")
print("key check ok", vocab, d_model, n_layers)
PY

ls -lh "$OUT"
