# RenderMatte_GRPO

`RenderMatte_GRPO` contains the reinforcement learning alignment stage for RenderMatte.

This folder is used after the SFT stage in `RenderMatte`. It starts from an existing SFT LoRA checkpoint, samples multiple candidate alpha mattes under the same trimap condition, computes matting rewards, and updates LoRA parameters with group-relative policy optimization.

The supervised training stage is not included here. Use `RenderMatte` for full-parameter SFT, LoRA SFT, and trimap inference and evaluation.

## Directory Layout

```text
config/grpo_matting.py            matting GRPO configs
scripts/train_flux_kontext.py     GRPO training entry point
scripts/train_rendermatte_grpo.sh     full GRPO launch wrapper
scripts/prepare_matting_metadata.py
scripts/accelerate_configs/       accelerate launch configs
rendermatte_grpo/rewards.py              matting reward
rendermatte_grpo/diffusers_patch/        sampling and log-prob utilities
rendermatte_grpo/ema.py                  EMA helper
rendermatte_grpo/stat_tracking.py        reward statistics helper
requirements.txt
```

## Environment

Use a FLUX.1 Kontext compatible training environment:

```bash
pip install -r requirements.txt
```

Quick check:

```bash
python - <<'PY'
import torch, accelerate, diffusers, transformers, peft, ml_collections
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Metadata Format

The prepared metadata directory is derived from `RenderMatte-dataset` and must contain:

```text
train_metadata.jsonl
test_metadata.jsonl
```

Each JSONL row should provide:

```json
{"prompt": "Transform to matting map while maintaining original composition", "image": "path/to/image.jpg", "trimap": "path/to/trimap.png", "alpha": "path/to/alpha.png"}
```

The GRPO stage requires trimap guidance for every sample.

## Prepare Metadata

For a split file with columns:

```text
image_path trimap_path alpha_path
```

run:

```bash
python scripts/prepare_matting_metadata.py \
  --split-file <DATA_ROOT>/RenderMatte-dataset/train/metadata.txt \
  --output-dir /path/to/matting_metadata \
  --split train \
  --root <DATA_ROOT>/RenderMatte-dataset

python scripts/prepare_matting_metadata.py \
  --split-file <DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k/filenames_RenderMatte-2k_relative.txt \
  --output-dir /path/to/matting_metadata \
  --split test \
  --root <DATA_ROOT>/RenderMatte-dataset
```

## Full GRPO Training

```bash
export MODEL_ROOT=/path/to/FLUX.1-Kontext-dev
export DATASET_DIR=/path/to/RenderMatte-dataset_metadata
export INIT_LORA_PATH=/path/to/sft_lora
export SAVE_DIR=/path/to/grpo_output
export CUDA_VISIBLE_DEVICES=0

export RESOLUTION=512
export NUM_STEPS=8
export EVAL_NUM_STEPS=1
export GUIDANCE_SCALE=1.0
export TRAIN_BATCH_SIZE=16
export TEST_BATCH_SIZE=2
export NUM_IMAGE_PER_PROMPT=16
export NUM_BATCHES_PER_EPOCH=8
export GRPO_BATCH_SIZE=16
export MAX_EPOCHS=50
export LEARNING_RATE=1e-6
export CLIP_RANGE=1e-4
export NOISE_LEVEL=0.2
export USE_EMA=1
export USE_8BIT_ADAM=1
export ACTIVATION_CHECKPOINTING=1
export USE_WANDB=0

bash scripts/train_rendermatte_grpo.sh
```

Checkpoints are saved as PEFT LoRA adapters:

```text
$SAVE_DIR/checkpoints/checkpoint-{step}/lora/
```

No LoRA conversion is performed in this stage.
