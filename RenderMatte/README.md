# RenderMatte

`RenderMatte` contains the supervised fine-tuning stage for trimap-guided alpha matting.

This folder is responsible for:

1. full-parameter SFT of the FLUX.1 Kontext DiT for matting,
2. optional LoRA SFT on top of the base model or a full-parameter SFT checkpoint,
3. trimap inference and benchmark evaluation for SFT or SFT+LoRA checkpoints.

The reinforcement learning stage is not included here. Use `RenderMatte_GRPO` for the subsequent GRPO-based alignment stage.

## Directory Layout

```text
configs/                          accelerate and model configs
scripts/train.py                  full-parameter and LoRA SFT training entry point
scripts/train_full_parameter_matting.sh
scripts/train_lora_matting.sh
scripts/infer_sft_lora_trimap.sh
inference_eval.py                 trimap inference and evaluation
matting_inference_core.py         shared inference utilities
models/                           FLUX.1 Kontext model components
pipelines/                        training and inference pipeline
lora/                             LoRA loading utilities
utils/                            matting metrics, losses, and samplers
prompters/                        prompt handling utilities
vram_management/                  memory management helpers
data_split/                       benchmark and training filename lists
requirements.txt
```

## Environment

Install the required packages in the training environment:

```bash
pip install -r requirements.txt
```

The scripts expect PyTorch, Accelerate, PEFT, Diffusers, Safetensors, and related runtime packages.

## Data Format

Training file lists use whitespace-separated relative paths. For matting SFT, the common format is:

```text
image_path trimap_path alpha_path
```

Paths are resolved relative to the `RenderMatte-dataset` root passed through environment variables or CLI arguments.

## Full-Parameter SFT Training

```bash
export PYTHON_BIN=/path/to/python
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/FLUX.1-Kontext-dev
export TRAIN_ROOT=<DATA_ROOT>/RenderMatte-dataset
export TRAIN_LIST=<DATA_ROOT>/RenderMatte-dataset/train/filenames_train.txt
export OUTPUT_DIR=/path/to/output/full_parameter_sft

bash scripts/train_full_parameter_matting.sh
```

Useful optional variables:

```bash
export HEIGHT=512
export WIDTH=512
export BATCH_SIZE=2
export LEARNING_RATE=1e-5
export NUM_EPOCHS=30
export SAVE_STEPS=2000
export NUM_WORKERS=0
```

Saved full-parameter checkpoints are written to `OUTPUT_DIR` as `step-*.safetensors` or `epoch-*.safetensors`.

## LoRA SFT Training

To train LoRA from the base model:

```bash
export PYTHON_BIN=/path/to/python
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/FLUX.1-Kontext-dev
export TRAIN_ROOT=<DATA_ROOT>/RenderMatte-dataset
export TRAIN_LIST=<DATA_ROOT>/RenderMatte-dataset/train/filenames_train.txt
export OUTPUT_DIR=/path/to/output/lora_sft

bash scripts/train_lora_matting.sh
```

To initialize from a full-parameter SFT checkpoint:

```bash
export INIT_DIT_CHECKPOINT=/path/to/full_parameter_sft_checkpoint.safetensors
bash scripts/train_lora_matting.sh
```

## Trimap Inference and Evaluation

Set benchmark roots:

```bash
export P3M_ROOT=/path/to/P3M-10k
export AM_ROOT=/path/to/AM-2k
export AIM_ROOT=/path/to/AIM-500
```

Run SFT-only evaluation:

```bash
export PYTHON_BIN=/path/to/python
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/FLUX.1-Kontext-dev
export SFT_CHECKPOINT=/path/to/full_parameter_sft_checkpoint.safetensors
export OUTPUT_ROOT=/path/to/eval_output
export BENCHMARKS="RenderMatte-2k"

bash scripts/infer_sft_lora_trimap.sh
```

Run SFT + LoRA evaluation:

```bash
export GRPO_LORA_PATH=/path/to/lora_checkpoint.safetensors
export LORA_RANK=64
export LORA_ALPHA=128
bash scripts/infer_sft_lora_trimap.sh
```

Default benchmark names are:

```text
p3m-np am aim RenderMatte-2k
```

The inference script writes raw predictions, trimap predictions, and a JSON summary under `OUTPUT_ROOT`.
