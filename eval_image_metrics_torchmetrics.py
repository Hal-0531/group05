import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

DEFAULT_SKIP_FIRST_N = 1
AVAILABLE_METRIC_NAMES = [
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "vif",
    "uiqi",
    "sam",
    "sdi",
]
DEFAULT_METRIC_NAMES = (
    "psnr",
    "ssim",
    "vif",
    "uiqi",
    "sam",
    "sdi",
)
DEFAULT_METRICS_CSV = ",".join(DEFAULT_METRIC_NAMES)
UNAVAILABLE_INIT_METRICS = set()
UNAVAILABLE_RUNTIME_METRICS = set()


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_frame(frame, target_size):
    if target_size is None:
        return frame
    if (frame.shape[1], frame.shape[0]) == target_size:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[0] > target_size[1] else cv2.INTER_LINEAR
    return cv2.resize(frame, target_size, interpolation=interpolation)


def _to_tensor(frame, device):
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


def build_metrics(metric_names, device):
    try:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
            MultiScaleStructuralSimilarityIndexMeasure,
            LearnedPerceptualImagePatchSimilarity,
            VisualInformationFidelity,
            UniversalImageQualityIndex,
            SpectralAngleMapper,
            SpectralDistortionIndex,
        )
    except Exception as exc:
        raise ImportError("Failed to import torchmetrics image metrics. Install torchmetrics[image].") from exc

    metrics = {}
    for name in metric_names:
        if name in UNAVAILABLE_INIT_METRICS:
            continue
        try:
            if name == "psnr":
                metrics[name] = PeakSignalNoiseRatio(data_range=1.0).to(device)
            elif name == "ssim":
                metrics[name] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            elif name == "ms_ssim":
                metrics[name] = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            elif name == "lpips":
                metrics[name] = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
            elif name == "vif":
                metrics[name] = VisualInformationFidelity().to(device)
            elif name == "uiqi":
                metrics[name] = UniversalImageQualityIndex().to(device)
            elif name == "sam":
                metrics[name] = SpectralAngleMapper().to(device)
            elif name == "sdi":
                metrics[name] = SpectralDistortionIndex().to(device)
            else:
                raise ValueError(f"Unknown metric: {name}. Available: {', '.join(AVAILABLE_METRIC_NAMES)}")
        except Exception as exc:
            print(f"[WARN] Skip metric '{name}': {exc}")
            UNAVAILABLE_INIT_METRICS.add(name)

    if not metrics:
        raise RuntimeError("No metrics could be initialized. Check dependencies and permissions.")
    return metrics


def eval_video_metrics_torchmetrics(
    ref_path,
    dist_path,
    device="auto",
    frame_stride=2,
    metric_names=None,
    skip_first_n=DEFAULT_SKIP_FIRST_N,
):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if metric_names is None:
        metric_names = list(DEFAULT_METRIC_NAMES)

    metrics = build_metrics(metric_names, device)
    for name in list(metrics.keys()):
        if name in UNAVAILABLE_RUNTIME_METRICS:
            metrics.pop(name, None)
    values = {name: [] for name in metrics.keys()}

    ref_cap = cv2.VideoCapture(ref_path)
    dist_cap = cv2.VideoCapture(dist_path)
    resized_once = False
    disabled_metrics = set()
    try:
        frame_idx = 0
        while True:
            ref_frame = read_frame(ref_cap)
            dist_frame = read_frame(dist_cap)
            if ref_frame is None or dist_frame is None:
                break
            if frame_idx < skip_first_n:
                frame_idx += 1
                continue
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            if ref_frame.shape != dist_frame.shape:
                if not resized_once:
                    print("Warning: frame sizes differ, resizing dist to ref size")
                    resized_once = True
                dist_frame = _resize_frame(dist_frame, (ref_frame.shape[1], ref_frame.shape[0]))

            ref = _to_tensor(ref_frame, device)
            dist = _to_tensor(dist_frame, device)

            for name, metric in list(metrics.items()):
                try:
                    if name == "lpips":
                        ref_lp = ref * 2.0 - 1.0
                        dist_lp = dist * 2.0 - 1.0
                        val = metric(ref_lp, dist_lp)
                    else:
                        val = metric(ref, dist)
                    values[name].append(float(val.detach().cpu().item()))
                except Exception as exc:
                    if name not in disabled_metrics:
                        print(f"[WARN] Disable metric '{name}' during evaluation: {exc}")
                        disabled_metrics.add(name)
                        UNAVAILABLE_RUNTIME_METRICS.add(name)
                    metrics.pop(name, None)
                    values.pop(name, None)

            frame_idx += 1
    finally:
        ref_cap.release()
        dist_cap.release()

    if not values:
        raise RuntimeError("No metrics remained after runtime checks.")

    frame_count = len(next(iter(values.values()))) if values else 0
    if frame_count == 0:
        raise ValueError(
            f"No frames evaluated. Check --stride ({frame_stride}) and fixed skip-first-n ({skip_first_n})."
        )

    result = {
        "frames": frame_count,
    }
    for name, vals in values.items():
        result[f"{name}_mean_over_t"] = sum(vals) / len(vals)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="video/preview.mp4")
    parser.add_argument("--dist", default="video/preview_imagined.mp4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS_CSV,
        help="Comma-separated list. Options: psnr,ssim,ms_ssim,lpips,vif,uiqi,sam,sdi (default excludes lpips/ms_ssim)",
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    result = eval_video_metrics_torchmetrics(
        args.ref,
        args.dist,
        device=args.device,
        frame_stride=args.stride,
        metric_names=metrics,
    )
    print(result)
