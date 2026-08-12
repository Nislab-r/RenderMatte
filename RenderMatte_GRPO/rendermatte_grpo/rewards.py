

from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict


def _load_matting_array(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    return arr


def _resize_matting_array(arr, height, width, resample=Image.BILINEAR):
    if arr.shape[:2] == (height, width):
        return arr
    image = Image.fromarray(arr.astype(np.float32))
    return np.array(image.resize((width, height), resample))


def _gauss(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _dgauss(x, sigma):
    return -x * _gauss(x, sigma) / (sigma**2)


def _gaussgradient(im, sigma):
    import scipy.ndimage

    epsilon = 1e-2
    halfsize = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))
    ).astype(np.int_)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = _gauss(u[0], sigma) * _dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()
    return (
        scipy.ndimage.convolve(im, hx, mode="nearest"),
        scipy.ndimage.convolve(im, hy, mode="nearest"),
    )


def _largest_cc(segmentation):
    from skimage.measure import label

    labels = label(segmentation, connectivity=1)
    counts = np.bincount(labels.flat)
    if len(counts) <= 1:
        return np.zeros_like(segmentation, dtype=bool)
    return labels == np.argmax(counts)


def _connectivity_error(pred, target, trimap=None, step=0.1):
    h, w = pred.shape
    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones((h, w), dtype=np.float32) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int_)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int_)
        omega = _largest_cc(pred_alpha_thresh * target_alpha_thresh).astype(np.int_)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int_)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1
    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int_)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int_)
    error = np.abs(pred_phi - target_phi)
    if trimap is not None:
        error = error[trimap == 128]
    return np.sum(error) / 1000.0


def _matting_metrics(pred, alpha, trimap):
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.nan_to_num(alpha.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    pred = np.clip(pred, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    if trimap is not None:
        if trimap.max() <= 1:
            trimap = (trimap * 255).round()
        trimap = trimap.astype(np.uint8)
        unknown = trimap == 128
        pred_eval = pred.copy()
        pred_eval[trimap == 255] = 1.0
        pred_eval[trimap == 0] = 0.0
    else:
        unknown = np.ones_like(pred, dtype=bool)
        pred_eval = pred

    pixel = float(unknown.sum())
    if pixel <= 0:
        pixel = float(pred.size)
        unknown = np.ones_like(pred, dtype=bool)

    abs_diff = np.abs(pred_eval - alpha)
    sq_diff = (pred_eval - alpha) ** 2
    mse = float(np.sum(sq_diff[unknown]) / pixel)
    mad = float(np.sum(abs_diff[unknown]) / pixel)
    sad = float(np.sum(abs_diff) / 1000.0)

    pred_x, pred_y = _gaussgradient(pred, 1.4)
    alpha_x, alpha_y = _gaussgradient(alpha, 1.4)
    grad_map = (np.sqrt(pred_x**2 + pred_y**2) - np.sqrt(alpha_x**2 + alpha_y**2)) ** 2
    grad = float(np.sum(grad_map[unknown]) / 1000.0)
    conn = float(_connectivity_error(pred, alpha, trimap))
    return mse, mad, sad, grad, conn, pixel


def _matting_metrics_e2p_aligned(pred, alpha, trimap):
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.nan_to_num(alpha.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    pred = np.clip(pred, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    if trimap is not None:
        if trimap.max() <= 1:
            trimap = (trimap * 255).round()
        trimap = trimap.astype(np.uint8)
        unknown = trimap == 128
        known = ~unknown

        pred_unknown_eval = pred.copy()
        pred_unknown_eval[trimap == 255] = 1.0
        pred_unknown_eval[trimap == 0] = 0.0
    else:
        unknown = np.ones_like(pred, dtype=bool)
        known = np.zeros_like(pred, dtype=bool)
        pred_unknown_eval = pred

    unknown_pixels = float(unknown.sum())
    if unknown_pixels <= 0:
        unknown = np.ones_like(pred, dtype=bool)
        unknown_pixels = float(pred.size)
    known_pixels = float(known.sum())

    abs_whole = np.abs(pred - alpha)
    sq_whole = (pred - alpha) ** 2
    abs_unknown = np.abs(pred_unknown_eval - alpha)
    sq_unknown = (pred_unknown_eval - alpha) ** 2

    pred_x, pred_y = _gaussgradient(pred, 1.4)
    alpha_x, alpha_y = _gaussgradient(alpha, 1.4)
    grad_map = (np.sqrt(pred_x**2 + pred_y**2) - np.sqrt(alpha_x**2 + alpha_y**2)) ** 2

    metrics = {
        # Match RenderMatte evaluation with whole-image matting metrics.
        "whole_mse": float(np.sum(sq_whole) / pred.size),
        "whole_mad": float(np.sum(abs_whole) / pred.size),
        "whole_sad": float(np.sum(abs_whole) / 1000.0),
        "whole_grad": float(np.sum(grad_map) / 1000.0),
        "whole_conn": float(_connectivity_error(pred, alpha, None)),
        # Keep trimap-sensitive boundary pressure as auxiliary signal.
        "unknown_mse": float(np.sum(sq_unknown[unknown]) / unknown_pixels),
        "unknown_mad": float(np.sum(abs_unknown[unknown]) / unknown_pixels),
        "unknown_grad": float(np.sum(grad_map[unknown]) / 1000.0),
        "unknown_conn": float(_connectivity_error(pred, alpha, trimap)),
        "unknown_pixels": unknown_pixels,
    }
    metrics["known_l1"] = (
        float(np.sum(abs_whole[known]) / known_pixels) if known_pixels > 0 else 0.0
    )
    return metrics


def matting_score(device):
    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = images.detach().float().cpu().clamp(0, 1).numpy()
            images = images.transpose(0, 2, 3, 1)

        scores = []
        details = defaultdict(list)
        # SAD-oriented reward for leaderboard tuning. Keep a few auxiliary
        # terms only as guardrails so SAD remains the dominant optimization
        # signal near the current SOTA regime.
        weights = {
            "whole_sad": 1.00,
            "whole_mad": 0.25,
            "whole_grad": 0.15,
            "whole_conn": 0.10,
            "known_l1": 0.20,
        }
        scales = {
            "whole_sad": 15.0,
            "whole_mad": 0.025,
            "whole_grad": 20.0,
            "whole_conn": 15.0,
            "known_l1": 0.015,
        }

        for image, meta in zip(images, metadata):
            alpha_path = meta.get("alpha") or meta.get("alpha_path") or meta.get("gt")
            trimap_path = meta.get("trimap") or meta.get("trimap_path")
            dataset_root = meta.get("__dataset_root", "")
            if not alpha_path:
                scores.append(-10.0)
                continue
            if not os.path.isabs(alpha_path):
                alpha_path = os.path.join(dataset_root, alpha_path)
            if trimap_path and not os.path.isabs(trimap_path):
                trimap_path = os.path.join(dataset_root, trimap_path)

            pred = image.mean(axis=2)
            alpha = _load_matting_array(alpha_path)
            alpha = _resize_matting_array(alpha, pred.shape[0], pred.shape[1])
            trimap = None
            if trimap_path:
                trimap = _load_matting_array(trimap_path)
                trimap = _resize_matting_array(trimap, pred.shape[0], pred.shape[1], Image.NEAREST)

            metric_values = _matting_metrics_e2p_aligned(pred, alpha, trimap)
            normalized_error = sum(
                weights[name] * min(metric_values[name] / scales[name], 10.0)
                for name in weights
            )
            reward = float(-normalized_error)
            scores.append(reward)

            for name, value in metric_values.items():
                if name == "unknown_pixels":
                    continue
                details[f"matting_{name}"].append(float(value))
                if name in weights:
                    details[f"matting_{name}_term"].append(
                        float(weights[name] * min(value / scales[name], 10.0))
                    )
            # Backward-compatible aliases used by train_flux_kontext.py summaries.
            details["matting_mse"].append(float(metric_values["whole_mse"]))
            details["matting_mad"].append(float(metric_values["whole_mad"]))
            details["matting_sad"].append(float(metric_values["whole_sad"]))
            details["matting_grad"].append(float(metric_values["whole_grad"]))
            details["matting_conn"].append(float(metric_values["whole_conn"]))
            details["matting_unknown_pixels"].append(float(metric_values["unknown_pixels"]))
            details["matting_error"].append(float(normalized_error))

        return scores, dict(details)

    return _fn

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "matting": matting_score,
        "jpeg_compressibility": jpeg_compressibility,
    }
    unknown = sorted(set(score_dict) - set(score_functions))
    if unknown:
        raise ValueError(f"Unsupported reward(s) in matting release: {unknown}")

    score_fns = {}
    for score_name in score_dict:
        fn = score_functions[score_name]
        if "device" in fn.__code__.co_varnames:
            score_fns[score_name] = fn(device)
        else:
            score_fns[score_name] = fn()

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        del ref_images, only_strict
        total_scores = []
        score_details = {}
        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            scores = np.asarray(scores, dtype=np.float32)
            total_scores.append(weight * scores)
            if isinstance(rewards, dict):
                for key, value in rewards.items():
                    score_details[key] = value
            score_details[score_name] = scores

        if not total_scores:
            raise ValueError("score_dict must contain at least one reward")
        score_details["avg"] = np.sum(total_scores, axis=0)
        return score_details, {}

    return _fn

if __name__ == "__main__":
    main()
