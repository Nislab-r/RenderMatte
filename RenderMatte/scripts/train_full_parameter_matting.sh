#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the FLUX.1-Kontext-dev model directory}"
: "${TRAIN_ROOT:?Set TRAIN_ROOT to the matting training dataset root}"
: "${TRAIN_LIST:?Set TRAIN_LIST to the training filename list}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"
PROMPT="${PROMPT:-Transform to matting map while maintaining original composition}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_LIST="${EVAL_LIST:-}"
EVAL_ROOT="${EVAL_ROOT:-}"
EVAL_DATASET="${EVAL_DATASET:-aim}"

cmd=(
  "$PYTHON_BIN" -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" scripts/train.py
  --dataset_base_path "$TRAIN_ROOT"
  --dataset_metadata_path "$TRAIN_LIST"
  --data_file_keys kontext_images,image
  --model_paths "$MODEL_ROOT"
  --learning_rate "$LEARNING_RATE"
  --num_epochs "$NUM_EPOCHS"
  --remove_prefix_in_ckpt pipe.dit.
  --trainable_models dit
  --extra_inputs kontext_images
  --multi_res_noise
  --default_caption "$PROMPT"
  --with_mask
  --batch_size "$BATCH_SIZE"
  --save_steps "$SAVE_STEPS"
  --matting_prompt trimap
  --task matting
  --output_path "$OUTPUT_DIR"
  --height "$HEIGHT"
  --width "$WIDTH"
  --adamw8bit
  --use_gradient_checkpointing
  --dataset_num_workers "$NUM_WORKERS"
)

if [[ -n "$EVAL_LIST" ]]; then
  : "${EVAL_ROOT:?Set EVAL_ROOT when EVAL_LIST is provided}"
  cmd+=(--eval_file_list "$EVAL_LIST" --eval_base_path "$EVAL_ROOT" --eval_dataset "$EVAL_DATASET")
fi

"${cmd[@]}"
