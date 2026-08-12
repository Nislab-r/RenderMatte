import os
from ml_collections import ConfigDict


def _env(name, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name, default):
    return int(_env(name, str(default)))


def _env_float(name, default):
    return float(_env(name, str(default)))


def _env_bool(name, default=False):
    value = _env(name, str(default)).lower()
    return value in {"1", "true", "yes", "y", "on"}


def matting_flux_kontext_grpo():
    config = ConfigDict()
    config.run_name = _env("RUN_NAME", "rendermatte_grpo")
    config.logdir = _env("LOGDIR", "logs")
    config.save_dir = _env("SAVE_DIR", "outputs/rendermatte_grpo")
    config.num_checkpoint_limit = _env_int("NUM_CHECKPOINT_LIMIT", 5)
    config.seed = _env_int("SEED", 42)
    config.mixed_precision = _env("MIXED_PRECISION", "bf16")
    config.allow_tf32 = _env_bool("ALLOW_TF32", True)
    config.resolution = _env_int("RESOLUTION", 512)
    config.use_lora = True
    config.guidance_embeds = False
    config.activation_checkpointing = _env_bool("ACTIVATION_CHECKPOINTING", True)
    config.condition_on_trimap = True
    config.per_prompt_stat_tracking = _env_bool("PER_PROMPT_STAT_TRACKING", False)
    config.eval_at_start = _env_bool("EVAL_AT_START", False)
    config.save_at_start = _env_bool("SAVE_AT_START", False)
    config.eval_freq = _env_int("EVAL_FREQ", 2)
    config.save_freq = _env_int("SAVE_FREQ", 2)
    config.max_epochs = _env_int("MAX_EPOCHS", 50)
    config.debug_save_images = _env_bool("DEBUG_SAVE_IMAGES", False)
    config.debug_max_images = _env_int("DEBUG_MAX_IMAGES", 16)
    config.use_wandb = _env_bool("USE_WANDB", False)
    config.wandb_project = _env("WANDB_PROJECT", "rendermatte_grpo")

    config.dataset = _env("DATASET_DIR", "dataset/rendermatte_grpo")
    config.pretrained = ConfigDict()
    config.pretrained.model = _env("MODEL_ROOT", "models/FLUX.1-Kontext-dev")

    config.sample = ConfigDict()
    config.sample.num_steps = _env_int("NUM_STEPS", 8)
    config.sample.eval_num_steps = _env_int("EVAL_NUM_STEPS", 1)
    config.sample.guidance_scale = _env_float("GUIDANCE_SCALE", 1.0)
    config.sample.eval_guidance_scale = _env_float("EVAL_GUIDANCE_SCALE", 1.0)
    config.sample.train_batch_size = _env_int("TRAIN_BATCH_SIZE", 16)
    config.sample.test_batch_size = _env_int("TEST_BATCH_SIZE", 2)
    config.sample.num_image_per_prompt = _env_int("NUM_IMAGE_PER_PROMPT", 16)
    config.sample.num_batches_per_epoch = _env_int("NUM_BATCHES_PER_EPOCH", 8)
    config.sample.noise_level = _env_float("NOISE_LEVEL", 0.2)
    config.sample.global_std = _env_bool("GLOBAL_STD", True)
    config.sample.same_latent = _env_bool("SAME_LATENT", False)

    config.train = ConfigDict()
    config.train.lora_path = _env("INIT_LORA_PATH", "")
    config.train.batch_size = _env_int("GRPO_BATCH_SIZE", config.sample.train_batch_size)
    config.train.gradient_accumulation_steps = _env_int("GRADIENT_ACCUMULATION_STEPS", 1)
    config.train.num_inner_epochs = _env_int("NUM_INNER_EPOCHS", 1)
    config.train.timestep_fraction = _env_float("TIMESTEP_FRACTION", 1.0)
    config.train.beta = _env_float("KL_BETA", 0.0)
    config.train.use_8bit_adam = _env_bool("USE_8BIT_ADAM", True)
    config.train.ema = _env_bool("USE_EMA", True)
    config.train.learning_rate = _env_float("LEARNING_RATE", 1e-6)
    config.train.adam_beta1 = _env_float("ADAM_BETA1", 0.9)
    config.train.adam_beta2 = _env_float("ADAM_BETA2", 0.999)
    config.train.adam_weight_decay = _env_float("ADAM_WEIGHT_DECAY", 0.0)
    config.train.adam_epsilon = _env_float("ADAM_EPSILON", 1e-8)
    config.train.clip_range = _env_float("CLIP_RANGE", 1e-4)
    config.train.adv_clip_max = _env_float("ADV_CLIP_MAX", 5.0)
    config.train.max_grad_norm = _env_float("MAX_GRAD_NORM", 1.0)

    config.reward_fn = {"matting": _env_float("MATTING_REWARD_WEIGHT", 1.0)}
    return config


def get_config(name="matting_flux_kontext_grpo"):
    if name == "matting_flux_kontext_grpo":
        return matting_flux_kontext_grpo()
    raise ValueError(f"Unknown config: {name}")
