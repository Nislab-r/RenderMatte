from collections import defaultdict
import contextlib
import os
import sys
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxKontextPipeline
from diffusers.utils.torch_utils import is_compiled_module
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
import numpy as np
import rendermatte_grpo.prompts
import rendermatte_grpo.rewards
from rendermatte_grpo.stat_tracking import PerPromptStatTracker
from rendermatte_grpo.diffusers_patch.flux_kontext_pipeline_with_logprob import pipeline_with_logprob
from rendermatte_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from rendermatte_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
try:
    import wandb
except Exception:
    wandb = None
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from rendermatte_grpo.ema import EMAModuleWrapper

import hashlib
import gc  # 🟢 新增：用于强制垃圾回收
from absl import app, flags

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def use_wandb(config):
    return bool(config.get("use_wandb", False)) and wandb is not None


def safe_wandb_log(config, *args, **kwargs):
    if use_wandb(config):
        wandb.log(*args, **kwargs)


def safe_wandb_image(config, *args, **kwargs):
    if use_wandb(config):
        return wandb.Image(*args, **kwargs)
    return None


class GenevalPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.dataset, path)
    
    def __getitem__(self, idx):
        metadata = dict(self.metadatas[idx])
        metadata["__dataset_root"] = self.dataset
        item = {
            "prompt": self.prompts[idx],
            "metadata": metadata
        }
        # Assuming 'image' in metadata contains a path to the image file
        # 1. 从元数据中获取图片相对路径
        image_path = self.metadatas[idx]['image']
        trimap_path = metadata.get("trimap") or metadata.get("trimap_path")
        # 2. 生成一个唯一的标识符（提示词 + 图片路径）
        # 这在后续用于生成确定的随机噪声种子（create_generator 函数会用到）
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}_{trimap_path}"
        # 3. 真正从磁盘读取图片
        # 使用 PIL 打开图片并强制转换为 RGB 模式（去除透明通道等干扰）
        image = Image.open(self._resolve_path(image_path)).convert('RGB')
        item["image"] = image
        item["trimap_image"] = None
        if trimap_path:
            item["trimap_image"] = Image.open(self._resolve_path(trimap_path)).convert("L")
        return item

    @staticmethod
    def collate_fn(examples):
        # examples 是一个列表，里面包含了 Batch Size 数量的字典
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        trimap_images = [example["trimap_image"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        # 返回四个列表，分别对应 Batch 里的所有 Prompt、元数据、图片和路径
        return prompts, metadatas, images, trimap_images, prompt_with_image_paths

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def prepare_e2p_trimap_condition(trimap_image, size):
    # e2p trimap convention: grayscale 0/128/255, nearest resize, repeated to 3 channels.
    return trimap_image.convert("L").resize(size, Image.Resampling.NEAREST).convert("RGB")

def build_kontext_condition_images(ref_images, trimap_images, config):
    if not config.get("condition_on_trimap", False):
        return ref_images
    if trimap_images is None or any(trimap is None for trimap in trimap_images):
        raise ValueError("condition_on_trimap=True requires every metadata row to provide a trimap path.")

    condition_groups = []
    for ref_image, trimap_image in zip(ref_images, trimap_images):
        trimap_image = prepare_e2p_trimap_condition(trimap_image, ref_image.size)
        condition_groups.append([ref_image, trimap_image])
    return {"multi_condition_images": condition_groups}

def summarize_reward_arrays(rewards):
    summary = {}
    for key, value in rewards.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr != -10]
        if arr.size == 0:
            continue
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
    return summary

def append_reward_history(save_dir, record):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "reward_history.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def print_reward_summary(phase, epoch, step, summary):
    preferred_keys = [
        "ori_avg",
        "avg",
        "matting",
        "matting_error",
        "matting_mse",
        "matting_mad",
        "matting_sad",
        "matting_grad",
        "matting_conn",
    ]
    parts = []
    for key in preferred_keys:
        if key in summary:
            stats = summary[key]
            parts.append(
                f"{key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}, n={stats['n']}"
            )
    if parts:
        print(f"\n[{phase} rewards] epoch={epoch} step={step}\n  " + "\n  ".join(parts), flush=True)

def save_debug_images(images, ref_images, metadata, save_dir, prefix, max_images=8):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(images, torch.Tensor):
        images = images.detach().float().cpu().clamp(0, 1).numpy()
        images = images.transpose(0, 2, 3, 1)

    count = min(max_images, len(images))
    for idx in range(count):
        sample_dir = os.path.join(save_dir, f"{prefix}_{idx:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        generated = Image.fromarray((images[idx] * 255).round().clip(0, 255).astype(np.uint8))
        generated.save(os.path.join(sample_dir, "generated.png"))

        if ref_images is not None:
            ref_images[idx].save(os.path.join(sample_dir, "input.png"))

        meta = metadata[idx]
        dataset_root = meta.get("__dataset_root", "")
        for key, filename in [("alpha", "alpha.png"), ("trimap", "trimap.png")]:
            path = meta.get(key) or meta.get(f"{key}_path")
            if not path:
                continue
            if not os.path.isabs(path):
                path = os.path.join(dataset_root, path)
            if os.path.exists(path):
                Image.open(path).save(os.path.join(sample_dir, filename))
        
def compute_log_prob(transformer, pipeline, sample, j, config):
    # 1. 取出当前 timestep 的 latent; 也就是这一批样本在第 j 步的当前状态。
    latents = sample["latents"][:, j]
    device = latents.device
    dtype = latents.dtype
    # 如果模型支持 guidance embedding，就构造 guidance
    # 如果模型支持 guidance embedding，就构造 guidance
    # 兼容单卡和多卡(DDP)模式
    unwrapped_transformer = transformer.module if hasattr(transformer, "module") else transformer
    
    if unwrapped_transformer.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    # if transformer.module.config.guidance_embeds:
    #     guidance = torch.tensor([config.sample.guidance_scale], device=device)
    #     guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # Predict the noise residual
    # 默认输入是当前 latent。
    # 如果有 image_latents，就把参考图像 latent 和当前 latent 拼在一起。
    latent_model_input = sample["latents"][:, j]
    if sample["image_latents"] is not None:
        latent_model_input = torch.cat([latent_model_input, sample["image_latents"]],dim=1)
    #  model_pred 在当前策略参数下，这个 timestep 我认为下一步应该往哪里走
    model_pred = transformer(
        hidden_states=latent_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids= sample["text_ids"][0],
        img_ids=sample["latent_ids"][0],
        return_dict=False,
    )[0]
    # 因为输入里可能拼了 image_latents，输出要裁回真正对应 diffusion latent 的维度。
    model_pred = model_pred[:, : sample["latents"][:, j].size(1)]
    # compute the log prob of next_latents given latents under the current model
    # “我不重新采样新轨迹，而是在旧轨迹上评估：当前 policy 认为从当前 latent 跳到这条旧轨迹里的下一 latent，有多大概率？”
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    '''
    log_prob(对数概率密度) 用于计算 Policy Loss（让模型多生成高分图）。
        这是最重要的输出。它计算的是在以 prev_sample_mean 为中心、std_dev_t 为半径的高斯分布中，
        观测到 prev_sample 的概率密度的对数。 如果这个值变大了，说明模型正在调整参数，使得生成“好样本”的可能性变高。
    
    prev_sample_mean (预测均值 $\mu_\theta$)含义：这是模型认为 $x_{t-1}$ 最应该出现的位置。
    它是去掉噪声后的“理想预测值”。训练作用：它主要用于计算 KL 散度（KL Loss）。通过比较训练模型和参考模型（
    Reference Model）的均值差异，可以约束模型不要飘得太远，保持生成的图像不“崩坏”。

    std_dev_t (标准差 $\sigma_t$)含义：代表了这一步去噪过程中的不确定性（噪声强度）。来源：由调度器（Scheduler）和配置的 noise_level 共同决定。
    训练作用：它是 Log Prob 计算的分母。噪声越大，单点概率密度的差异就越不明显，梯度更新就越平滑。

    prev_sample (计算出的前一步样本)含义：在训练函数中，这个值通常是用来验证或作为中间变量。虽然函数会根据当前模型重新推算一个 
    $x_{t-1}$，但在计算 Loss 时，我们依然使用采样阶段存下来的那个 sample["next_latents"]。

    prev_sample_mean 和 std_dev_t 组合用于计算 KL Loss（让模型别乱跑）。
    '''

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for eval_batch_idx, test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        )):
        prompts, prompt_metadata, ref_images, trimap_images, _ = test_batch
        ref_images = [ref_image.resize((config.resolution, config.resolution)) for ref_image in ref_images]
        trimap_images = [
            trimap_image.resize((config.resolution, config.resolution), Image.Resampling.NEAREST)
            if trimap_image is not None
            else None
            for trimap_image in trimap_images
        ]
        condition_images = build_kontext_condition_images(ref_images, trimap_images, config)
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    image=condition_images,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    max_area=config.resolution*config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        debug_eval_batches = config.get("debug_eval_batches", 4)
        if (
            config.get("debug_save_images", False)
            and accelerator.is_main_process
            and eval_batch_idx < debug_eval_batches
        ):
            save_debug_images(
                images,
                ref_images,
                prompt_metadata,
                os.path.join(config.save_dir, "debug_images"),
                f"eval_step_{global_step}_batch_{eval_batch_idx}",
                max_images=config.get("debug_max_images", 8),
            )

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).float().cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().long().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).float().cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        eval_summary = summarize_reward_arrays(all_rewards)
        print_reward_summary("eval", None, global_step, eval_summary)
        append_reward_history(
            config.save_dir,
            {
                "phase": "eval",
                "global_step": int(global_step),
                "summary": eval_summary,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.float32).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            safe_wandb_log(config, 
                {
                    "eval_images": [
                        safe_wandb_image(config, 
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process and use_wandb(config):
        wandb.init(
            project=config.get("wandb_project", "rendermatte_grpo"),
            name=config.run_name,
            config=config.to_dict(),
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    pipeline = FluxKontextPipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=inference_dtype,
        low_cpu_mem_usage=True,
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)
    if config.activation_checkpointing and hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
        pipeline.transformer.enable_gradient_checkpointing()

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    train_dataset = GenevalPromptImageDataset(config.dataset, 'train')
    test_dataset = GenevalPromptImageDataset(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=GenevalPromptImageDataset.collate_fn,
        # persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=GenevalPromptImageDataset.collate_fn,
        shuffle=False,
        num_workers=0,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    # prepare prompt and reward fn
    if is_deepspeed_zero3_enabled():
        # Using deepspeed zero3 will cause the model parameter `weight.shape` to be empty.
        unset_hf_deepspeed_config()
        reward_fn = getattr(rendermatte_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(rendermatte_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        set_hf_deepspeed_config(accelerator.state.deepspeed_plugin.dschf)
    else:
        reward_fn = getattr(rendermatte_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(rendermatte_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=1) 

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    if config.get("eval_at_start", False):
        pipeline.transformer.eval()
        eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
    if config.get("save_at_start", False) and accelerator.is_main_process:
        save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch > 0 and epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch > 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata, ref_images, trimap_images, prompt_with_image_paths = next(train_iter)
            ref_images = [ref_image.resize((config.resolution, config.resolution)) for ref_image in ref_images]
            trimap_images = [
                trimap_image.resize((config.resolution, config.resolution), Image.Resampling.NEAREST)
                if trimap_image is not None
                else None
                for trimap_image in trimap_images
            ]
            condition_images = build_kontext_condition_images(ref_images, trimap_images, config)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            # the input of edit task is determined by both the image and the edit prompt
            prompt_ids = tokenizers[0](
                prompt_with_image_paths,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    """
                    images 生成的图片
                    latents 去噪过程中的每一个中间状态
                    latent_ids 输入图片的id和control img的ig concat的结果
                    text_ids 
                    log_probs 去噪过程中的每一个中间状态采样时候的概率
                    """
                    images, latents, latent_ids, text_ids, log_probs, image_latents = pipeline_with_logprob(
                        pipeline,
                        image=condition_images,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        max_area=config.resolution*config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator
                    )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)
            if config.get("debug_save_images", False) and accelerator.is_main_process:
                save_debug_images(
                    images,
                    ref_images,
                    prompt_metadata,
                    os.path.join(config.save_dir, "debug_images"),
                    f"train_epoch_{epoch}_batch_{i}",
                    max_images=config.get("debug_max_images", 8),
                )
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "latent_ids": latent_ids.unsqueeze(0).repeat(len(prompt_ids),1,1),
                    "image_latents": image_latents,
                    "text_ids": text_ids.unsqueeze(0).repeat(len(prompt_ids),1,1),
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )


        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                safe_wandb_log(config, 
                    {
                        "images": [
                            safe_wandb_image(config, 
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        '''
        假设batch size为4 
        1. samples["rewards"]["avg"] -> shape [4] 每个数值代表一张完整生成的图片的最终得分（由奖励模型打分）
        2. unsqueeze(1) shape -> [4, 1] 在列的方向增加了一个维度，此时依然只有总分
        3. repeat(1, 50) shape => [4, 50] 将那一个总分，在时间维度上复制了 50 次。

        在强化学习（RL）中，通常每个动作（Action）都应该对应一个奖励。对于扩散模型而言，
        生成一张图需要经历 $T$ 个时间步（去噪动作）。但问题是：奖励模型（Reward Model）
        通常只能给最后生成的“成品图”打分，它不知道第 5 步去噪做得好不好，只知道最后的结果好不好。
        通过这个操作，代码实际上是在假设：“如果这张图最后拿了高分，那么生成它的每一个去噪步骤（Step）都立了功，
        都应该分到这个高分。”

        因为后面训练时是对每个 timestep 单独算 policy ratio 和 loss。
        所以 reward/advantage 也必须对齐到时间维。
        '''
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        '''
        为什么要 gather？

        因为 advantage 计算不能只看单卡上的一小撮样本。
        特别是 per-prompt stat tracking 时，需要全局看到同 prompt 的多次采样结果。
        '''
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.float().cpu().numpy() for key, value in gathered_rewards.items()}

        train_reward_summary = summarize_reward_arrays(gathered_rewards)
        if accelerator.is_local_main_process:
            print("\n[Debug] Raw Rewards: ", gathered_rewards["ori_avg"])
            print_reward_summary("train", epoch, global_step, train_reward_summary)
        # log rewards and images
        if accelerator.is_main_process:
            append_reward_history(
                config.save_dir,
                {
                    "phase": "train",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "summary": train_reward_summary,
                },
            )
            safe_wandb_log(config, 
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            # 把所有 GPU 上的 prompt_ids 收集起来，再 decode 成字符串 prompt。
            # prompt_ids = accelerator.gather(samples["prompt_ids"]).float().cpu().numpy()
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().long().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                safe_wandb_log(config, 
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            safe_wandb_log(config, 
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        # 因为 rollout 很贵，所以收集到一批旧轨迹后，会对它重复优化多次。
        # inner_epoch（内部迭代） 是指利用同一批采样的旧数据，进行多次重复的参数更新。
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            # 程序把 samples reshape 成若干个小 batch，并转换成 list of dicts 方便遍历
            '''
            这时每个训练 batch 里已经包含：
                一组旧轨迹
                对应条件
                old log_probs
                advantages
            '''
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        '''
                        对一个训练 batch 来说，程序不是一次性把整条轨迹扔进 loss。
                            而是：

                            先枚举 train_timesteps
                            对每个 timestep 单独计算当前策略在该步上的 log_prob
                            再算 PPO/GRPO 损失

                            所以这里的真实执行顺序是：
                            batch 内部，再按时间步逐步更新。
                        '''
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                            # 如果开了 KL 正则，程序还会再算一个 reference policy
                            # “我不仅想提高 reward，也不想让 RL 后的 LoRA 离原始基模偏得太离谱。”
                            # 如果开了 KL 正则，程序还会再算一个 reference policy
                            # “我不仅想提高 reward，也不想让 RL 后的 LoRA 离原始基模偏得太离谱。”
                            if config.train.beta > 0:
                                unwrapped_transformer = transformer.module if hasattr(transformer, "module") else transformer
                                with torch.no_grad():
                                    with unwrapped_transformer.disable_adapter():
                                        prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref = compute_log_prob(transformer, pipeline, sample, j, config)
                            # if config.train.beta > 0:
                            #     with torch.no_grad():
                            #         with transformer.module.disable_adapter():
                            #             prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref = compute_log_prob(transformer, pipeline, sample, j, config)

                        # grpo logic
                        # 这里开始真正计算损失
                        # 取当前 timestep 的 advantage，并裁剪 避免极端 advantage 过大导致训练不稳定。
                        '''
                        有时候某张图的得分极其离谱（比如奖励模型出现幻觉，给了一个极高的分），这会导致优势值爆炸，
                        一次更新就把模型彻底带偏。截断是为了防止这种“极端个例”对模型产生过大影响。
                        '''
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        # ratio 当前策略相对于旧策略，对这个“动作”（latent transition）更偏爱还是更不偏爱。
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        # 计算未裁剪损失
                        # 如果 advantage 正，希望 ratio 变大；
                        # 如果 advantage 负，希望 ratio 变小。
                        unclipped_loss = -advantages * ratio
                        # 更新可以，但别一步跨太大。
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        # 取两者最大值得到 policy_loss
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        # 如果开了 KL，就再加上 beta * kl_loss
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                    #   每个 timestep 的 loss 算完后，程序记录监控指标 这些不直接参与优化，但用来诊断训练有没有跑飞。
                        # info["approx_kl"].append(
                        #     0.5
                        #     * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        # )
                        # info["clipfrac"].append(
                        #     torch.mean(
                        #         (
                        #             torch.abs(ratio - 1.0) > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        # info["clipfrac_gt_one"].append(
                        #     torch.mean(
                        #         (
                        #             ratio - 1.0 > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        # info["clipfrac_lt_one"].append(
                        #     torch.mean(
                        #         (
                        #             1.0 - ratio > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        # info["policy_loss"].append(policy_loss)
                        # if config.train.beta > 0:
                        #     info["kl_loss"].append(kl_loss)

                        # info["loss"].append(loss)

                        info["approx_kl"].append(
                            (0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)).detach()
                        )
                        info["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()).detach()
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean((ratio - 1.0 > config.train.clip_range).float()).detach()
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean((1.0 - ratio > config.train.clip_range).float()).detach()
                        )
                        info["policy_loss"].append(policy_loss.detach())
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss.detach())

                        info["loss"].append(loss.detach())

                        # backward pass
                        # 然后做真正的反向传播和参数更新
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            safe_wandb_log(config, info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        epoch+=1
        if config.get("max_epochs", None) is not None and epoch >= config.max_epochs:
            if accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            break
        
if __name__ == "__main__":
    app.run(main)
