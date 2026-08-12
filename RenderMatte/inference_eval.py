#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from matting_inference_core import (
    DEFAULT_BENCHMARKS,
    load_full_finetuned_weights,
    load_sft_transformer_checkpoint,
    load_grpo_lora_weights,
    read_file_list,
    resolve_data_path,
    resolve_dtype,
    run_benchmark_inference,
)
from models.unified_dataset import UnifiedDataset
from models.utils import parse_flux_model_configs
from pipelines.flux_image_new import FluxImagePipeline
from utils.eval_matting import test as eval_matting


def normalize_alpha(prediction):
    arr = np.asarray(prediction)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3].mean(axis=-1)
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[0] if arr.shape[0] == 1 else arr[:3].mean(axis=0)
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def load_trimap(path, size_hw):
    trimap = Image.open(path).convert("L")
    height, width = size_hw
    if trimap.size != (width, height):
        trimap = trimap.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(trimap)


def parse_subdir_overrides(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected BENCHMARK=SUBDIR, got {value!r}")
        benchmark, subdir = value.split("=", 1)
        benchmark = benchmark.strip()
        subdir = subdir.strip().strip("/")
        if not benchmark or not subdir:
            raise ValueError(f"Expected non-empty BENCHMARK=SUBDIR, got {value!r}")
        overrides[benchmark] = subdir
    return overrides


def hard_clamp_dataset(pred_root, out_root, benchmark_name, input_subdirs, output_subdirs, max_samples=None):
    cfg = DEFAULT_BENCHMARKS[benchmark_name]
    pred_root = Path(pred_root)
    out_root = Path(out_root)
    gt_root = Path(resolve_data_path(cfg.gt_root))
    samples = read_file_list(Path(resolve_data_path(cfg.file_list)), max_samples=max_samples)

    input_subdir = input_subdirs.get(benchmark_name, cfg.output_subdir)
    output_subdir = output_subdirs.get(benchmark_name, input_subdir)
    out_dir = out_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    failed = []
    done = 0

    for merged_rel, trimap_rel, _ in samples:
        pred_path = pred_root / input_subdir / Path(merged_rel).with_suffix(".npy")
        trimap_path = gt_root / trimap_rel
        save_path = out_dir / Path(merged_rel).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not pred_path.exists():
            missing.append(str(pred_path))
            continue

        try:
            alpha = normalize_alpha(np.load(pred_path))
            trimap = load_trimap(trimap_path, alpha.shape)
            alpha[trimap <= 10] = 0.0
            alpha[trimap >= 245] = 1.0
            np.save(save_path, alpha.astype(np.float32))
            done += 1
        except Exception as exc:
            failed.append({"sample": merged_rel, "error": repr(exc)})

    report = {
        "benchmark": benchmark_name,
        "input": str(pred_root / input_subdir),
        "output": str(out_dir),
        "done": done,
        "missing": len(missing),
        "failed": len(failed),
        "missing_examples": missing[:10],
        "failed_examples": failed[:10],
    }
    print(json.dumps(report, indent=2))
    return report


def evaluate_dataset(out_root, benchmark_name, output_subdirs, max_samples=None):
    cfg = DEFAULT_BENCHMARKS[benchmark_name]
    output_subdir = output_subdirs.get(benchmark_name, cfg.output_subdir)
    args = argparse.Namespace(
        pred_path=str(Path(out_root) / output_subdir),
        gt_path=resolve_data_path(cfg.gt_root),
        dataset=benchmark_name,
        max_samples=max_samples,
    )
    return eval_matting(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full-parameter matting inference with trimap evaluation."
    )
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--sft_checkpoint", default=None, help="Full-parameter SFT checkpoint loaded into pipe.dit.")
    parser.add_argument("--checkpoint", default=None, help="Alias of --sft_checkpoint for backward compatibility.")
    parser.add_argument("--grpo_lora_path", default=None, help="GRPO/RL LoRA checkpoint loaded on top of the SFT checkpoint.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_target_modules", default=None, help="Optional comma-separated LoRA target modules.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["p3m-np", "am", "aim"],
        choices=sorted(DEFAULT_BENCHMARKS.keys()),
    )
    parser.add_argument(
        "--prompt",
        default="Transform to matting map while maintaining original composition",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_weight_coverage", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_hard_clamp", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--deterministic_flow", action="store_true")
    parser.add_argument("--input_subdirs", nargs="*", default=[])
    parser.add_argument("--output_subdirs", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    raw_root = output_root / "raw_predictions"
    trimap_root = output_root / "trimap_predictions"
    raw_root.mkdir(parents=True, exist_ok=True)
    trimap_root.mkdir(parents=True, exist_ok=True)

    input_subdirs = parse_subdir_overrides(args.input_subdirs)
    output_subdirs = parse_subdir_overrides(args.output_subdirs)
    benchmarks = [DEFAULT_BENCHMARKS[name] for name in args.benchmarks]

    sft_checkpoint = args.sft_checkpoint or args.checkpoint
    if not args.skip_inference and not sft_checkpoint:
        raise ValueError("Please provide --sft_checkpoint for the full-parameter SFT weights.")

    load_info = {"status": "skip_inference"}
    inference_reports = []
    clamp_reports = []
    eval_reports = []

    if not args.skip_inference:
        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
        torch_dtype = resolve_dtype(args.dtype)
        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=args.device,
            model_configs=parse_flux_model_configs(args.model_root),
            model_base_path=args.model_root,
        )
        if str(sft_checkpoint).endswith("diffusion_pytorch_model.safetensors"):
            sft_load_info = load_sft_transformer_checkpoint(
                pipe=pipe,
                checkpoint_path=sft_checkpoint,
            )
        else:
            sft_load_info = load_full_finetuned_weights(
                pipe=pipe,
                checkpoint_path=sft_checkpoint,
                min_coverage=args.min_weight_coverage,
            )
        load_info = {"sft_full_parameter": sft_load_info, "grpo_lora": None}
        if args.grpo_lora_path:
            target_modules = None
            if args.lora_target_modules:
                target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
            load_info["grpo_lora"] = load_grpo_lora_weights(
                pipe=pipe,
                lora_path=args.grpo_lora_path,
                torch_dtype=torch_dtype,
                device=args.device,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
            )
        transform = UnifiedDataset.default_image_operator(
            height=args.resolution,
            width=args.resolution,
        )
    else:
        pipe = None
        transform = None

    for benchmark in benchmarks:
        if not args.skip_inference:
            inference_reports.append(
                run_benchmark_inference(
                    pipe=pipe,
                    benchmark=benchmark,
                    output_root=raw_root,
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
            )

        if not args.skip_hard_clamp:
            clamp_reports.append(
                hard_clamp_dataset(
                    pred_root=raw_root,
                    out_root=trimap_root,
                    benchmark_name=benchmark.name,
                    input_subdirs=input_subdirs,
                    output_subdirs=output_subdirs,
                    max_samples=args.max_samples,
                )
            )

        if not args.skip_evaluation:
            eval_reports.append(
                {
                    "benchmark": benchmark.name,
                    "metrics": evaluate_dataset(
                        trimap_root,
                        benchmark.name,
                        output_subdirs,
                        max_samples=args.max_samples,
                    ),
                }
            )

    summary = {
        "sft_checkpoint": sft_checkpoint,
        "grpo_lora_path": args.grpo_lora_path,
        "model_root": args.model_root,
        "raw_prediction_root": str(raw_root),
        "trimap_root": str(trimap_root),
        "weight_load": load_info,
        "inference": inference_reports,
        "trimap": clamp_reports,
        "evaluation": eval_reports,
    }
    with (output_root / "trimap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {output_root / 'trimap_summary.json'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
