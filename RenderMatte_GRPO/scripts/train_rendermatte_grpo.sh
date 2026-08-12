#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the FLUX.1 Kontext base model directory}"
: "${DATASET_DIR:?Set DATASET_DIR to a directory containing train_metadata.jsonl and test_metadata.jsonl}"
: "${INIT_LORA_PATH:?Set INIT_LORA_PATH to the SFT LoRA directory to continue from}"
: "${SAVE_DIR:?Set SAVE_DIR to the output directory}"

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  scripts/train_flux_kontext.py \
  --config config/grpo_matting.py:matting_flux_kontext_grpo
