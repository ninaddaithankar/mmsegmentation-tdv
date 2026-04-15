#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="configs/tdv/tdv-small_upernet_160k_ade20k-512x512.py"
# TDV_CKPT="/shared/nas2/ninadd2/tdv-checkpoints/FULL_INI1K_scratch_only_ssv2_2025-11-02_22-00-32_/last.ckpt"
# TDV_CKPT="/shared/nas2/ninadd2/tdv-new/logs/checkpoints/UNMASKED_finevideo_best_hparams_2026-02-09_16-05-08_/last.ckpt"
# TDV_CKPT="/shared/nas2/ninadd2/tdv-checkpoints/GRID_DS_PCT;ds=ssv2;hparams=prev_best;pct=0.01_2026-03-24_00-09-49_/last.ckpt"
# TDV_CKPT="/shared/nas2/ninadd2/tdv-checkpoints/GRID_DS_PCT;ds=ssv2;hparams=prev_best;pct=0.1_2026-03-24_00-09-48_/last.ckpt"
TDV_CKPT="/shared/nas2/ninadd2/tdv-checkpoints/VIT_SMALL|ds=ssv2|hparams=prev_best|pct=1.0_2026-03-28_11-55-32_/epoch=epoch=18-step=step=200602-valid/loss=valid/loss=7.5833.ckpt"
WORK_DIR="${1:-work_dirs/tdv-vit_s-ssv2-ade20k-fixed-encoder}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmseg-tdv}"
export WANDB_NAME="${WANDB_NAME:-$(basename "${WORK_DIR}")}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=5 python3 tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options model.backbone.checkpoint_path="${TDV_CKPT}" \
    train_cfg.max_iters=320000 \
    param_scheduler.1.end=320000 \
    default_hooks.checkpoint.interval=16000
