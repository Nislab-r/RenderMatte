#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from models.unified_dataset import UnifiedDataset
from models.flux_dit import FluxDiT
from models.utils import DiffusionTrainingModule, load_state_dict, parse_flux_model_configs
from pipelines.flux_image_new import FluxImagePipeline
from utils.eval_matting import test as builtin_eval_matting


# ============================================================
# Benchmark 配置
# ============================================================

@dataclass
class BenchmarkConfig:
    name: str
    gt_root: str
    file_list: str
    output_subdir: str


def resolve_data_path(path: str) -> str:
    data_root = os.environ.get(
        "DATA_ROOT",
        "/nvmedata/workspace2/users/rzc/datasets",
    )
    return str(path).replace("<DATA_ROOT>", data_root)


DEFAULT_BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # P3M-500-NP
    "p3m-np": BenchmarkConfig(
        name="p3m-np",
        gt_root=os.environ.get("P3M_ROOT", "<P3M_ROOT>"),
        file_list="./data_split/P3M_matting/filenames_val_NP.txt",
        output_subdir="p3m_np_full",
    ),

    # AM-2K
    "am": BenchmarkConfig(
        name="am",
        gt_root=os.environ.get("AM_ROOT", "<AM_ROOT>"),
        file_list="./data_split/AM_matting/filenames_val.txt",
        output_subdir="am_2k_full",
    ),

    # AIM-500
    "aim": BenchmarkConfig(
        name="aim",
        gt_root=os.environ.get("AIM_ROOT", "<AIM_ROOT>"),
        file_list="./data_split/AIM_matting/filenames_val.txt",
        output_subdir="aim_500_full",
    ),

    "RenderMatte-2k": BenchmarkConfig(
        name="RenderMatte-2k",
        gt_root="<DATA_ROOT>/RenderMatte-dataset",
        file_list="<DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k/filenames_RenderMatte-2k_relative.txt",
        output_subdir="RenderMatte-2k_full",
    ),

    "RenderMatte-2k_hard_tree_flower": BenchmarkConfig(
        name="RenderMatte-2k_hard_tree_flower",
        gt_root="<DATA_ROOT>/RenderMatte-dataset",
        file_list="<DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k_hard_tree_flower/filenames_RenderMatte-2k_hard_tree_flower_relative.txt",
        output_subdir="RenderMatte-2k_hard_tree_flower_full",
    ),

    "RenderMatte-2k_mild_hard": BenchmarkConfig(
        name="RenderMatte-2k_mild_hard",
        gt_root="<DATA_ROOT>/RenderMatte-dataset",
        file_list="<DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k_mild_hard/filenames_RenderMatte-2k_mild_hard_relative.txt",
        output_subdir="RenderMatte-2k_mild_hard_full",
    ),

    "RenderMatte-2k_mild_harder": BenchmarkConfig(
        name="RenderMatte-2k_mild_harder",
        gt_root="<DATA_ROOT>/RenderMatte-dataset",
        file_list="<DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k_mild_harder/filenames_RenderMatte-2k_mild_harder_relative.txt",
        output_subdir="RenderMatte-2k_mild_harder_full",
    ),
    "RenderMatte-2k_sad8_target": BenchmarkConfig(
        name="RenderMatte-2k_sad8_target",
        gt_root="<DATA_ROOT>/RenderMatte-dataset",
        file_list="<DATA_ROOT>/RenderMatte-dataset/RenderMatte-2k_sad8_target/filenames_RenderMatte-2k_sad8_target_relative.txt",
        output_subdir="RenderMatte-2k_sad8_target_full",
    ),
}


# ============================================================
# 全参数微调权重加载
# ============================================================

def unwrap_state_dict(checkpoint):
    """
    从常见 checkpoint 格式中提取真正的 state_dict。
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "module",
        "dit",
        "transformer",
        "net",
        "network",
    ]

    for key in possible_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            print(f"[weights] 使用 checkpoint['{key}'] 作为 state_dict")
            return value

    return checkpoint


def remove_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """
    仅保留带指定前缀的参数，然后移除前缀。
    """
    output = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            output[key[len(prefix):]] = value

    return output


def strip_common_prefixes(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    移除所有参数共同拥有的通用前缀。
    """
    prefixes = [
        "module.",
        "_orig_mod.",
    ]

    output = state_dict

    changed = True
    while changed:
        changed = False

        for prefix in prefixes:
            if output and all(key.startswith(prefix) for key in output):
                output = {
                    key[len(prefix):]: value
                    for key, value in output.items()
                }
                changed = True

    return output


def load_single_weight_file(
    weight_path: Path,
) -> Dict[str, torch.Tensor]:
    suffix = weight_path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "加载 .safetensors 需要安装 safetensors："
                "pip install safetensors"
            ) from exc

        state_dict = load_file(str(weight_path), device="cpu")
    else:
        checkpoint = torch.load(
            str(weight_path),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = unwrap_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"权重文件没有返回字典格式 state_dict：{weight_path}"
        )

    return strip_common_prefixes(state_dict)


def load_sharded_safetensors(
    checkpoint_dir: Path,
    index_file: Path,
) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "加载分片 safetensors 需要安装 safetensors。"
        ) from exc

    with index_file.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))

    if not shard_names:
        raise RuntimeError(
            f"没有在索引文件中找到 weight_map：{index_file}"
        )

    state_dict = {}

    for shard_name in tqdm(
        shard_names,
        desc="Loading safetensors shards",
    ):
        shard_path = checkpoint_dir / shard_name

        if not shard_path.exists():
            raise FileNotFoundError(
                f"索引中记录的分片不存在：{shard_path}"
            )

        shard_state = load_file(str(shard_path), device="cpu")
        state_dict.update(shard_state)

    return strip_common_prefixes(state_dict)


def load_sharded_pytorch(
    checkpoint_dir: Path,
    index_file: Path,
) -> Dict[str, torch.Tensor]:
    with index_file.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))

    if not shard_names:
        raise RuntimeError(
            f"没有在索引文件中找到 weight_map：{index_file}"
        )

    state_dict = {}

    for shard_name in tqdm(
        shard_names,
        desc="Loading PyTorch shards",
    ):
        shard_path = checkpoint_dir / shard_name

        if not shard_path.exists():
            raise FileNotFoundError(
                f"索引中记录的分片不存在：{shard_path}"
            )

        shard_checkpoint = torch.load(
            str(shard_path),
            map_location="cpu",
            weights_only=False,
        )
        shard_state = unwrap_state_dict(shard_checkpoint)

        if not isinstance(shard_state, dict):
            raise TypeError(
                f"分片没有返回 state_dict：{shard_path}"
            )

        state_dict.update(shard_state)

    return strip_common_prefixes(state_dict)


def find_weight_source(checkpoint_path: Path):
    """
    checkpoint_path 可以是单个文件或 checkpoint 目录。
    """
    if checkpoint_path.is_file():
        return "single", checkpoint_path

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"找不到 checkpoint：{checkpoint_path}"
        )

    safetensors_indexes = [
        checkpoint_path / "model.safetensors.index.json",
        checkpoint_path / "diffusion_pytorch_model.safetensors.index.json",
        checkpoint_path / "pytorch_model.safetensors.index.json",
    ]

    for index_path in safetensors_indexes:
        if index_path.exists():
            return "sharded_safetensors", index_path

    pytorch_indexes = [
        checkpoint_path / "pytorch_model.bin.index.json",
        checkpoint_path / "diffusion_pytorch_model.bin.index.json",
    ]

    for index_path in pytorch_indexes:
        if index_path.exists():
            return "sharded_pytorch", index_path

    preferred_names = [
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
        "model.pt",
        "model.pth",
        "pytorch_model.bin",
        "checkpoint.pt",
        "checkpoint.pth",
    ]

    for name in preferred_names:
        candidate = checkpoint_path / name
        if candidate.exists():
            return "single", candidate

    candidate_files = []

    for pattern in [
        "*.safetensors",
        "*.pt",
        "*.pth",
        "*.bin",
    ]:
        candidate_files.extend(checkpoint_path.glob(pattern))

    candidate_files = sorted(candidate_files)

    if len(candidate_files) == 1:
        return "single", candidate_files[0]

    if not candidate_files:
        raise FileNotFoundError(
            f"目录中未找到可加载的模型权重：{checkpoint_path}"
        )

    raise RuntimeError(
        "checkpoint 目录中存在多个候选权重，无法自动判断：\n"
        + "\n".join(str(x) for x in candidate_files)
        + "\n请通过 --checkpoint 指定具体权重文件。"
    )


def read_checkpoint_state_dict(
    checkpoint_path: str,
) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    source_type, source_path = find_weight_source(checkpoint_path)

    print(f"[weights] source_type={source_type}")
    print(f"[weights] source_path={source_path}")

    if source_type == "single":
        return load_single_weight_file(source_path)

    if source_type == "sharded_safetensors":
        return load_sharded_safetensors(
            checkpoint_path,
            source_path,
        )

    if source_type == "sharded_pytorch":
        return load_sharded_pytorch(
            checkpoint_path,
            source_path,
        )

    raise ValueError(f"不支持的权重来源类型：{source_type}")


def build_candidate_state_dicts(
    state_dict: Dict[str, torch.Tensor],
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    构造不同前缀版本，自动寻找最适合 pipe.dit 的形式。

    全参数训练脚本经常把模型参数保存为：
        dit.xxx
        model.dit.xxx
        module.dit.xxx
        pipe.dit.xxx
        diffusion_model.xxx
        transformer.xxx
    """
    candidate_prefixes = [
        "",
        "dit.",
        "model.dit.",
        "module.dit.",
        "pipe.dit.",
        "pipeline.dit.",
        "diffusion_model.",
        "model.diffusion_model.",
        "transformer.",
        "model.transformer.",
        "model.",
        "module.",
    ]

    candidates = []
    seen_key_sets = set()

    for prefix in candidate_prefixes:
        if prefix == "":
            candidate = state_dict
            name = "raw"
        else:
            candidate = remove_prefix_from_state_dict(
                state_dict,
                prefix,
            )
            name = f"remove_prefix={prefix}"

        if not candidate:
            continue

        signature = tuple(sorted(candidate.keys())[:100])

        if signature in seen_key_sets:
            continue

        seen_key_sets.add(signature)
        candidates.append((name, candidate))

    return candidates


def count_matching_parameters(
    model_state_dict: Dict[str, torch.Tensor],
    candidate_state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, int, int]:
    """
    返回：
        匹配 key 数量
        shape 匹配参数量
        shape 不匹配 key 数量
    """
    matched_keys = 0
    matched_numel = 0
    shape_mismatches = 0

    for key, value in candidate_state_dict.items():
        if key not in model_state_dict:
            continue

        matched_keys += 1

        if tuple(value.shape) == tuple(model_state_dict[key].shape):
            matched_numel += value.numel()
        else:
            shape_mismatches += 1

    return matched_keys, matched_numel, shape_mismatches


def select_best_state_dict_for_dit(
    dit: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[str, Dict[str, torch.Tensor]]:
    model_state = dit.state_dict()
    candidates = build_candidate_state_dicts(state_dict)

    scored_candidates = []

    for name, candidate in candidates:
        matched_keys, matched_numel, shape_mismatches = (
            count_matching_parameters(model_state, candidate)
        )

        scored_candidates.append(
            (
                matched_numel,
                matched_keys,
                -shape_mismatches,
                name,
                candidate,
            )
        )

        print(
            f"[weights] candidate={name}, "
            f"matched_keys={matched_keys}, "
            f"matched_numel={matched_numel:,}, "
            f"shape_mismatches={shape_mismatches}"
        )

    if not scored_candidates:
        raise RuntimeError("没有构造出可用的 state_dict 候选。")

    scored_candidates.sort(
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True,
    )

    _, matched_keys, _, best_name, best_state_dict = scored_candidates[0]

    if matched_keys == 0:
        checkpoint_examples = list(state_dict.keys())[:20]
        model_examples = list(model_state.keys())[:20]

        raise RuntimeError(
            "checkpoint 中没有任何参数能够匹配 pipe.dit。\n"
            f"checkpoint keys 示例：{checkpoint_examples}\n"
            f"pipe.dit keys 示例：{model_examples}"
        )

    return best_name, best_state_dict



LORA_TARGET_MODULES = [
    "a_to_qkv",
    "b_to_qkv",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "a_to_out",
    "b_to_out",
    "proj_out",
    "norm.linear",
    "norm1_a.linear",
    "norm1_b.linear",
    "to_qkv_mlp",
]


def load_sft_transformer_checkpoint(
    pipe: FluxImagePipeline,
    checkpoint_path: str,
):
    """Load the original stage-1 SFT transformer checkpoint in diffusers format."""
    print(f"[weights] Loading SFT DiT checkpoint: {checkpoint_path}")
    state = load_file(str(checkpoint_path), device="cpu")
    converted = FluxDiT.state_dict_converter().from_diffusers(state)
    if isinstance(converted, tuple):
        converted = converted[0]
    result = pipe.dit.load_state_dict(converted, strict=False)
    if hasattr(result, "missing_keys"):
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
    else:
        missing = list(result[0])
        unexpected = list(result[1])
    print(
        f"[weights] SFT DiT converted: keys={len(converted)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print("[weights] missing SFT key sample:")
        for key in missing[:20]:
            print(f"  {key}")
    if unexpected:
        print("[weights] unexpected SFT key sample:")
        for key in unexpected[:20]:
            print(f"  {key}")
    pipe.dit.eval()
    return {
        "path": str(checkpoint_path),
        "format": "diffusers_transformer",
        "converted_keys": len(converted),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def load_grpo_lora_weights(
    pipe: FluxImagePipeline,
    lora_path: str,
    torch_dtype,
    device: str,
    lora_rank: int = 64,
    lora_alpha: Optional[int] = None,
    target_modules: Optional[Sequence[str]] = None,
):
    """Attach and load the GRPO/RL LoRA adapter on top of the SFT DiT weights."""
    from lora.flux_lora import FluxLoRALoader

    target_modules = list(target_modules or LORA_TARGET_MODULES)
    helper = DiffusionTrainingModule()
    pipe.dit = helper.add_lora_to_model(
        pipe.dit,
        target_modules=target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        upcast_dtype=pipe.torch_dtype,
    )

    state_dict = load_state_dict(lora_path)
    loader = FluxLoRALoader(torch_dtype=torch_dtype, device=device)
    state_dict = loader.convert_state_dict(state_dict)
    state_dict = helper.mapping_lora_state_dict(state_dict)

    load_result = pipe.dit.load_state_dict(state_dict, strict=False)
    if hasattr(load_result, "missing_keys"):
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
    else:
        missing = list(load_result[0])
        unexpected = list(load_result[1])
    lora_missing = [key for key in missing if "lora_" in key]

    print(
        "[weights] GRPO LoRA loaded: "
        f"path={lora_path}, keys={len(state_dict)}, "
        f"missing={len(missing)}, missing_lora={len(lora_missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if unexpected:
        print("[weights] unexpected LoRA key sample:")
        for key in unexpected[:20]:
            print(f"  {key}")
    if lora_missing:
        print("[weights] missing LoRA key sample:")
        for key in lora_missing[:20]:
            print(f"  {key}")

    pipe.dit.eval()
    return {
        "path": str(lora_path),
        "rank": lora_rank,
        "alpha": lora_alpha if lora_alpha is not None else lora_rank,
        "target_modules": target_modules,
        "loaded_keys": len(state_dict),
        "missing_keys": len(missing),
        "missing_lora_keys": len(lora_missing),
        "unexpected_keys": len(unexpected),
    }


def load_full_finetuned_weights(
    pipe: FluxImagePipeline,
    checkpoint_path: str,
    min_coverage: float = 0.90,
):
    """
    将全参数微调权重直接加载到 pipe.dit。

    与 LoRA 不同：
    - 不创建 PEFT adapter
    - 不进行 LoRA state_dict 转换
    - 不调用 mapping_lora_state_dict
    - 直接加载完整 DiT 参数
    """
    print(f"[weights] 正在读取全参数微调权重：{checkpoint_path}")

    checkpoint_state = read_checkpoint_state_dict(checkpoint_path)

    best_name, candidate_state = select_best_state_dict_for_dit(
        pipe.dit,
        checkpoint_state,
    )

    print(f"[weights] 选择参数映射方式：{best_name}")

    model_state = pipe.dit.state_dict()
    filtered_state = {}
    shape_mismatch_keys = []

    for key, value in candidate_state.items():
        if key not in model_state:
            continue

        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch_keys.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(model_state[key].shape),
                )
            )
            continue

        filtered_state[key] = value

    total_model_numel = sum(
        value.numel()
        for value in model_state.values()
    )
    loaded_numel = sum(
        value.numel()
        for value in filtered_state.values()
    )
    coverage = loaded_numel / max(total_model_numel, 1)

    print(
        f"[weights] matched_tensors={len(filtered_state)}/"
        f"{len(model_state)}"
    )
    print(
        f"[weights] parameter_coverage={coverage:.4%} "
        f"({loaded_numel:,}/{total_model_numel:,})"
    )

    if shape_mismatch_keys:
        print(
            f"[weights] shape 不匹配参数数量："
            f"{len(shape_mismatch_keys)}"
        )
        for item in shape_mismatch_keys[:20]:
            print(
                f"  {item[0]}: checkpoint={item[1]}, "
                f"model={item[2]}"
            )

    if coverage < min_coverage:
        raise RuntimeError(
            f"全参数权重覆盖率只有 {coverage:.2%}，"
            f"低于要求的 {min_coverage:.2%}。\n"
            "这通常表示 checkpoint 前缀、模型配置或加载目标不正确。"
        )

    load_result = pipe.dit.load_state_dict(
        filtered_state,
        strict=False,
    )

    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    print(
        f"[weights] missing={len(missing_keys)}, "
        f"unexpected={len(unexpected_keys)}"
    )

    if missing_keys:
        print("[weights] missing key 示例：")
        for key in missing_keys[:20]:
            print(f"  {key}")

    if unexpected_keys:
        print("[weights] unexpected key 示例：")
        for key in unexpected_keys[:20]:
            print(f"  {key}")

    pipe.dit.eval()

    return {
        "mapping": best_name,
        "coverage": coverage,
        "loaded_tensors": len(filtered_state),
        "model_tensors": len(model_state),
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }


# ============================================================
# 数据路径和输出处理
# ============================================================

def parse_file_list_line(
    line: str,
) -> Tuple[str, str, str]:
    """
    支持：

    三列：
        merged_path trimap_path alpha_path

    多列：
        第一列作为 merged
        第二列作为 trimap
        最后一列作为 alpha
    """
    items = line.strip().split()

    if len(items) < 2:
        raise ValueError(
            f"文件列表行至少需要两列：{line!r}"
        )

    merged_path = items[0]
    trimap_path = items[1]
    alpha_path = items[-1]

    return merged_path, trimap_path, alpha_path


def read_file_list(
    file_list_path: Path,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    if not file_list_path.exists():
        raise FileNotFoundError(
            f"找不到文件列表：{file_list_path}"
        )

    samples = []

    with file_list_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                samples.append(parse_file_list_line(line))
            except Exception as exc:
                raise ValueError(
                    f"解析文件列表失败："
                    f"{file_list_path}:{line_number}\n"
                    f"内容：{line}"
                ) from exc

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def prediction_relative_path(
    merged_relative_path: str,
) -> Path:
    """
    保留输入图像的相对目录结构，只把扩展名替换为 .npy。
    """
    return Path(merged_relative_path).with_suffix(".npy")


def normalize_prediction_array(
    prediction,
) -> np.ndarray:
    """
    将 pipeline 输出统一成二维 float32 alpha matte。

    支持：
        [B, H, W, C]
        [B, C, H, W]
        [H, W, C]
        [C, H, W]
        [H, W]
    """
    if torch.is_tensor(prediction):
        prediction = prediction.detach().float().cpu().numpy()

    prediction = np.asarray(prediction)

    while prediction.ndim > 3 and prediction.shape[0] == 1:
        prediction = prediction[0]

    if prediction.ndim == 4:
        prediction = prediction[0]

    if prediction.ndim == 3:
        # HWC
        if prediction.shape[-1] in (1, 3, 4):
            if prediction.shape[-1] == 1:
                prediction = prediction[..., 0]
            else:
                prediction = prediction[..., :3].mean(axis=-1)

        # CHW
        elif prediction.shape[0] in (1, 3, 4):
            if prediction.shape[0] == 1:
                prediction = prediction[0]
            else:
                prediction = prediction[:3].mean(axis=0)

        else:
            raise ValueError(
                f"无法判断三维输出的通道维度："
                f"shape={prediction.shape}"
            )

    if prediction.ndim != 2:
        raise ValueError(
            f"期望二维 alpha matte，实际 shape={prediction.shape}"
        )

    prediction = prediction.astype(np.float32)

    prediction[np.isnan(prediction)] = 0.0
    prediction[np.isposinf(prediction)] = 1.0
    prediction[np.isneginf(prediction)] = 0.0

    # 如果 pipeline 输出是 0~255，则转成 0~1。
    if prediction.size > 0 and prediction.max() > 1.5:
        prediction = prediction / 255.0

    prediction = np.clip(prediction, 0.0, 1.0)

    return prediction


# ============================================================
# 推理
# ============================================================

def run_benchmark_inference(
    pipe: FluxImagePipeline,
    benchmark: BenchmarkConfig,
    output_root: Path,
    transform,
    prompt: str,
    resolution: int,
    seed: int,
    num_inference_steps: int,
    cfg_scale: float,
    deterministic_flow: bool,
    max_samples: Optional[int],
    overwrite: bool,
) -> Dict:
    gt_root = Path(resolve_data_path(benchmark.gt_root))
    file_list = Path(resolve_data_path(benchmark.file_list))
    benchmark_output = output_root / benchmark.output_subdir

    benchmark_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = read_file_list(
        file_list,
        max_samples=max_samples,
    )

    print()
    print("=" * 80)
    print(f"[inference] benchmark={benchmark.name}")
    print(f"[inference] gt_root={gt_root}")
    print(f"[inference] file_list={file_list}")
    print(f"[inference] output={benchmark_output}")
    print(f"[inference] samples={len(samples)}")
    print("=" * 80)

    succeeded = 0
    skipped = 0
    failed = []

    for index, sample in enumerate(
        tqdm(samples, desc=f"Inference: {benchmark.name}"),
        1,
    ):
        merged_rel, trimap_rel, _ = sample

        image_path = gt_root / merged_rel
        trimap_path = gt_root / trimap_rel
        save_path = benchmark_output / prediction_relative_path(
            merged_rel
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if save_path.exists() and not overwrite:
            try:
                existing = np.load(
                    save_path,
                    mmap_mode="r",
                )

                if existing.size > 0:
                    skipped += 1
                    continue
            except Exception:
                print(
                    f"[inference] 检测到损坏输出，重新推理："
                    f"{save_path}"
                )

        if not image_path.exists():
            failed.append(
                {
                    "sample": merged_rel,
                    "error": f"image not found: {image_path}",
                }
            )
            continue

        if not trimap_path.exists():
            failed.append(
                {
                    "sample": merged_rel,
                    "error": f"trimap not found: {trimap_path}",
                }
            )
            continue

        try:
            kontext_images = [
                transform(str(image_path)),
                transform(str(trimap_path)),
            ]

            # Keep the inference path aligned with test.py: save the raw pipeline
            # output directly and leave shape/range handling to evaluation.
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    kontext_images=kontext_images,
                    height=resolution,
                    width=resolution,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    output_type="np",
                    rand_device=pipe.device,
                    deterministic_flow=False,
                    task="matting",
                )

            temporary_path = save_path.with_suffix(
                save_path.suffix + ".tmp"
            )

            # 使用文件句柄避免 np.save 自动添加 .npy 后缀。
            with temporary_path.open("wb") as f:
                np.save(f, output)

            os.replace(temporary_path, save_path)
            succeeded += 1

        except Exception as exc:
            failed.append(
                {
                    "sample": merged_rel,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"\n[inference][ERROR] {merged_rel}: {exc}"
            )

        if index % 20 == 0:
            print(
                f"[inference] {index}/{len(samples)}, "
                f"new={succeeded}, skipped={skipped}, "
                f"failed={len(failed)}"
            )

    failure_file = benchmark_output / "inference_failures.json"

    with failure_file.open("w", encoding="utf-8") as f:
        json.dump(
            failed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "benchmark": benchmark.name,
        "output_path": str(benchmark_output),
        "total": len(samples),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(failed),
        "failure_file": str(failure_file),
    }


# ============================================================
# Matting 指标评估
# ============================================================

def parse_builtin_eval_result(result_text: str) -> Dict[str, float]:
    """
    Parse the comma-separated output returned by utils.eval_matting.test().
    Expected order: MSE, MAD, SAD, Grad, Conn.
    """
    values = [float(x.strip()) for x in result_text.strip().split(",")]
    if len(values) != 5:
        raise ValueError(f"无法解析内置评估结果：{result_text!r}")
    return {
        "MSE": values[0],
        "MAD": values[1],
        "SAD": values[2],
        "Grad": values[3],
        "Conn": values[4],
    }


def run_benchmark_evaluation(
    benchmark: BenchmarkConfig,
    benchmark_output: Path,
    max_samples: Optional[int],
    workers: int,
) -> Dict:
    """
    Use the repository's built-in matting evaluation path instead of keeping a
    separate metric implementation in this script.
    """
    gt_root = Path(benchmark.gt_root)

    print()
    print("=" * 80)
    print(f"[eval] benchmark={benchmark.name}")
    print(f"[eval] prediction_root={benchmark_output}")
    print(f"[eval] gt_root={gt_root}")
    print("[eval] backend=utils.eval_matting.test")
    print("=" * 80)

    eval_args = argparse.Namespace(
        pred_path=str(benchmark_output),
        gt_path=str(gt_root),
        dataset=benchmark.name,
        max_samples=max_samples,
    )
    result_text = builtin_eval_matting(eval_args)
    parsed = parse_builtin_eval_result(result_text)

    # utils.eval_matting.test() already prints the exact valid count. Keep the
    # summary fields consistent for downstream JSON/TXT logging.
    total = len(read_file_list(Path(benchmark.file_list), max_samples=max_samples))
    result = {
        "benchmark": benchmark.name,
        "total": total,
        "valid": total,
        "failed": 0,
        **parsed,
        "error_file": "builtin_eval_matting",
    }

    return result

# ============================================================
# 汇总
# ============================================================

def print_summary(results: Sequence[Dict]):
    print()
    print("=" * 94)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 94)
    print(
        f"{'Benchmark':<15}"
        f"{'Valid':>10}"
        f"{'MSE':>13}"
        f"{'MAD':>13}"
        f"{'SAD':>13}"
        f"{'Grad':>13}"
        f"{'Conn':>13}"
    )
    print("-" * 94)

    for result in results:
        valid_text = (
            f"{result['valid']}/{result['total']}"
        )

        print(
            f"{result['benchmark']:<15}"
            f"{valid_text:>10}"
            f"{result['MSE']:>13.4f}"
            f"{result['MAD']:>13.4f}"
            f"{result['SAD']:>13.4f}"
            f"{result['Grad']:>13.4f}"
            f"{result['Conn']:>13.4f}"
        )

    print("=" * 94)


def save_summary(
    output_root: Path,
    args,
    load_info: Dict,
    inference_results: Sequence[Dict],
    evaluation_results: Sequence[Dict],
):
    summary = {
        "checkpoint": args.checkpoint,
        "model_root": args.model_root,
        "prompt": args.prompt,
        "resolution": args.resolution,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "deterministic_flow": args.deterministic_flow,
        "weight_load": load_info,
        "inference": list(inference_results),
        "evaluation": list(evaluation_results),
    }

    json_path = output_root / "benchmark_results.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    text_path = output_root / "benchmark_results.txt"

    with text_path.open("w", encoding="utf-8") as f:
        f.write(
            f"{'Benchmark':<15}"
            f"{'Valid':>10}"
            f"{'MSE':>13}"
            f"{'MAD':>13}"
            f"{'SAD':>13}"
            f"{'Grad':>13}"
            f"{'Conn':>13}\n"
        )

        for result in evaluation_results:
            valid_text = (
                f"{result['valid']}/{result['total']}"
            )

            f.write(
                f"{result['benchmark']:<15}"
                f"{valid_text:>10}"
                f"{result['MSE']:>13.4f}"
                f"{result['MAD']:>13.4f}"
                f"{result['SAD']:>13.4f}"
                f"{result['Grad']:>13.4f}"
                f"{result['Conn']:>13.4f}\n"
            )

    print(f"[summary] JSON：{json_path}")
    print(f"[summary] TXT ：{text_path}")


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "使用全参数微调后的 RenderMatte 模型，"
            "对多个 matting benchmark 进行推理和评估。"
        )
    )

    parser.add_argument(
        "--model_root",
        type=str,
        default="<MODEL_ROOT>/FLUX.1-Kontext-dev",
        help="基础模型目录。",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "全参数微调权重文件或 checkpoint 目录。"
            "不执行 LoRA 转换。"
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="三个 benchmark 的统一输出目录。",
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["p3m-np", "am", "aim"],
        choices=sorted(DEFAULT_BENCHMARKS.keys()),
        help="需要推理和评估的 benchmark。",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Transform to matting map while maintaining original composition"
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="评估阶段的 CPU 进程数量。",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个 benchmark 最多运行多少个样本。",
    )

    parser.add_argument(
        "--min_weight_coverage",
        type=float,
        default=0.90,
        help="完整权重加载的最低参数覆盖率。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 .npy 推理结果。",
    )

    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="跳过推理，仅评估 output_root 下已有结果。",
    )

    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="只推理，不计算指标。",
    )

    parser.add_argument(
        "--deterministic_flow",
        action="store_true",
        help="将 deterministic_flow 设置为 True。",
    )

    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


# ============================================================
# 主函数
# ============================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError(
            "当前环境没有可用 CUDA，但 --device 指定了 CUDA。"
        )

    output_root = Path(args.output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_benchmarks = [
        DEFAULT_BENCHMARKS[name]
        for name in args.benchmarks
    ]

    print("[config] benchmarks:")
    for benchmark in selected_benchmarks:
        print(json.dumps(
            asdict(benchmark),
            ensure_ascii=False,
        ))

    torch_dtype = resolve_dtype(args.dtype)

    pipe = None
    load_info = {}

    if not args.skip_inference:
        print("[load] 正在加载基础 pipeline")

        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=args.device,
            model_configs=parse_flux_model_configs(
                args.model_root
            ),
            model_base_path=args.model_root,
        )

        print("[load] 正在加载全参数微调 checkpoint")

        load_info = load_full_finetuned_weights(
            pipe=pipe,
            checkpoint_path=args.checkpoint,
            min_coverage=args.min_weight_coverage,
        )

        transform = UnifiedDataset.default_image_operator(
            height=args.resolution,
            width=args.resolution,
        )
    else:
        transform = None
        load_info = {
            "status": "skip_inference",
        }

    inference_results = []
    evaluation_results = []

    for benchmark in selected_benchmarks:
        benchmark_output = (
            output_root / benchmark.output_subdir
        )

        if not args.skip_inference:
            inference_result = run_benchmark_inference(
                pipe=pipe,
                benchmark=benchmark,
                output_root=output_root,
                transform=transform,
                prompt=args.prompt,
                resolution=args.resolution,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                deterministic_flow=args.deterministic_flow,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
            )

            inference_results.append(
                inference_result
            )

        if not args.skip_evaluation:
            evaluation_result = run_benchmark_evaluation(
                benchmark=benchmark,
                benchmark_output=benchmark_output,
                max_samples=args.max_samples,
                workers=args.workers,
            )

            evaluation_results.append(
                evaluation_result
            )

    if evaluation_results:
        print_summary(evaluation_results)
    save_summary(
        output_root=output_root,
        args=args,
        load_info=load_info,
        inference_results=inference_results,
        evaluation_results=evaluation_results,
    )

    print()
    print("所有 benchmark 推理和评估已完成。")


if __name__ == "__main__":
    # spawn 多进程必须放在 main 保护下。
    mp.freeze_support()
    main()
