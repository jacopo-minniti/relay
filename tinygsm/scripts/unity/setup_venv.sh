#!/bin/bash
# Build the TinyGSM conda prefix env. Run on a compute node, not the login node:
#   salloc -p cpu -c 4 --mem=16G -t 01:00:00 -- srun bash -lc './scripts/unity/setup_venv.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$PWD/.conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

if [ ! -x .venv_relay/bin/python ]; then
  conda create -p ./.venv_relay python=3.11 pip -y
fi
conda activate ./.venv_relay
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
python -c "import torch, relay_tinygsm, mlm, xlm; print('venv-ok', torch.__version__, torch.version.cuda)"
