#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ROOT:?Set MODEL_ROOT to the FLUX.1-Kontext-dev model directory}"
: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the full-parameter SFT checkpoint}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the inference/evaluation output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GRPO_LORA_PATH="${GRPO_LORA_PATH:-}"
BENCHMARKS="${BENCHMARKS:-p3m-np am aim RenderMatte-2k}"
RESOLUTION="${RESOLUTION:-1024}"
WORKERS="${WORKERS:-32}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda:0}"

cmd=(
  "$PYTHON_BIN" inference_eval.py
  --model_root "$MODEL_ROOT"
  --sft_checkpoint "$SFT_CHECKPOINT"
  --output_root "$OUTPUT_ROOT"
  --benchmarks $BENCHMARKS
  --resolution "$RESOLUTION"
  --num_inference_steps 1
  --cfg_scale 1
  --device "$DEVICE"
  --dtype bf16
  --lora_rank "$LORA_RANK"
  --workers "$WORKERS"
)

if [[ -n "$LORA_ALPHA" ]]; then
  cmd+=(--lora_alpha "$LORA_ALPHA")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max_samples "$MAX_SAMPLES")
fi

if [[ -n "$GRPO_LORA_PATH" ]]; then
  cmd+=(--grpo_lora_path "$GRPO_LORA_PATH")
fi

"${cmd[@]}"
