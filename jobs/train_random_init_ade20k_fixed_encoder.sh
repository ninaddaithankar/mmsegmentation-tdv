#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/tdv/tdv-base_upernet_160k_ade20k-512x512.py"
WORK_DIR="${1:-work_dirs/random-init-ade20k-fixed-encoder}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmseg-tdv}"
export WANDB_NAME="${WANDB_NAME:-$(basename "${WORK_DIR}")}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=2 python3 tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options model.backbone.random_init=True \
    train_cfg.max_iters=320000 \
    param_scheduler.1.end=320000 \
    default_hooks.checkpoint.interval=16000
