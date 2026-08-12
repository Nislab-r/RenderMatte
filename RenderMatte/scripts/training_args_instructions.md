# Training Arguments Explanation

This document provides a detailed explanation of the training arguments used in our scripts.

### Core Arguments
*   `--model_paths`: Path to the base model directory (e.g., `./FLUX.1-Kontext-dev`).
*   `--output_path`: Directory to save model checkpoints and logs.
*   `--learning_rate`: The learning rate for the optimizer (e.g., `1e-5`).
*   `--num_epochs`: Total number of training epochs.
*   `--batch_size`: Per-GPU batch size. Adjust based on your GPU memory.
*   `--resume`: Resume training from the latest checkpoint in `--output_path`.

### Dataset Arguments
*   `--dataset_base_path`: Comma-separated absolute paths to the training datasets. **The order matters.**
*   `--dataset_metadata_path`: Path to the metadata file (CSV or Parquet) containing relative file paths.
*   `--data_file_keys`: Column names in the metadata file to be used, e.g., `kontext_images,image`.
*   `--dataset_repeat`: Number of times to repeat a dataset within an epoch. Useful for balancing datasets of different sizes.
*   `--height`, `--width`: Target resolution for training images.

### Model & Training Strategy
*   `--trainable_models`: Specifies which parts of the model to train. For fine-tuning, set to `"dit"`.
*   `--extra_inputs`: Specifies additional input keys besides the main image and prompt. In our case, it's `"kontext_images"`.
*   `--default_caption`: The default text prompt used for training (e.g., `"Transform to normal map..."`).
*   `--multi_res_noise`: (Flag) Use multi-resolution noise for potentially faster convergence, inspired by Marigold.
*   `--with_mask`: (Flag) Compute the loss only on valid masked areas (e.g., where ground truth depth is available).
*   `--using_sqrt`: (Flag, Depth only) Use our theoretically optimal square-root normalization for depth.
*   `--extra_loss`: Name of the pixel-space consistency loss to apply (e.g., `"cycle_consistency_normal_estimation"`).
*   `--deterministic_flow`: (Flag) Use a fixed random seed for the initial noise to create a pseudo-deterministic path.

### LoRA Specific Arguments
*   `--lora_base_model`: The base model to which LoRA is applied, typically `"dit"`.
*   `--lora_target_modules`: Comma-separated list of modules to apply LoRA to.
*   `--lora_rank`: The rank of the LoRA decomposition matrices (e.g., `64`).
*   `--align_to_opensource_format`: (Flag) Save LoRA weights in a community-standard format.

### Performance & Logging
*   `--use_gradient_checkpointing`: (Flag) Enable to save GPU memory at the cost of a small slowdown.
*   `--adamw8bit`: (Flag) Use the 8-bit AdamW optimizer to reduce memory usage.
*   `--save_steps`: Save a full model checkpoint every N steps.
*   `--eval_steps`: Perform a quick evaluation on a small subset every N steps to monitor progress.
*   `--eval_file_list`: Path to the text file containing the list of images for evaluation during training.