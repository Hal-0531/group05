import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


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
            raise ValueError(f"Unknown metric: {name}")
    return metrics


def eval_video_metrics_torchmetrics(ref_path, dist_path, device="cpu", frame_stride=2, metric_names=None):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if metric_names is None:
        metric_names = ["psnr", "ssim"]

    metrics = build_metrics(metric_names, device)
    values = {name: [] for name in metric_names}

    ref_cap = cv2.VideoCapture(ref_path)
    dist_cap = cv2.VideoCapture(dist_path)
    resized_once = False
    try:
        frame_idx = 0
        while True:
            ref_frame = read_frame(ref_cap)
            dist_frame = read_frame(dist_cap)
            if ref_frame is None or dist_frame is None:
                break
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

            for name, metric in metrics.items():
                if name == "lpips":
                    ref_lp = ref * 2.0 - 1.0
                    dist_lp = dist * 2.0 - 1.0
                    val = metric(ref_lp, dist_lp)
                else:
                    val = metric(ref, dist)
                values[name].append(float(val.detach().cpu().item()))

            frame_idx += 1
    finally:
        ref_cap.release()
        dist_cap.release()

    frame_count = len(next(iter(values.values()))) if values else 0
    if frame_count == 0:
        raise ValueError("No frames read from one or both videos")

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
        default="psnr,ssim",
        help="Comma-separated list. Options: psnr,ssim,ms_ssim,lpips,vif,uiqi,sam,sdi",
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
