#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/tdv/tdv-base_upernet_160k_ade20k-512x512.py"
TDV_CKPT="/shared/nas2/ninadd2/tdv-checkpoints/FULL_INI1K_scratch_only_ssv2_2025-11-02_22-00-32_/last.ckpt"
WORK_DIR="${1:-work_dirs/tdv-ade20k-fixed-encoder}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmseg-tdv}"
export WANDB_NAME="${WANDB_NAME:-$(basename "${WORK_DIR}")}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options model.backbone.checkpoint_path="${TDV_CKPT}"
