#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/tdv/dino-base_upernet_160k_ade20k-512x512.py"
DINO_CKPT="/shared/nas2/ninadd2/repos/dino-recipe/output/FULL_AUG_dino_ssv2_only_/checkpoint.pth"
WORK_DIR="${1:-work_dirs/dino-ade20k-fixed-encoder}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmseg-tdv}"
export WANDB_NAME="${WANDB_NAME:-$(basename "${WORK_DIR}")}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=2 python3 tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume \
  --cfg-options model.backbone.checkpoint_path="${DINO_CKPT}" \
    train_cfg.max_iters=320000 \
    param_scheduler.1.end=320000 \
    default_hooks.checkpoint.interval=16000
