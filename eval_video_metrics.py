import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from DISTS.DISTS_pytorch.DISTS_pt import DISTS


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


def load_video_frames(path, frame_stride=2, target_size=None):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        idx = 0
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue
            frame = _resize_frame(frame, target_size)
            frames.append(frame)
            idx += 1
    finally:
        cap.release()
    return frames


def eval_videos(ref_path, dist_path, device="cpu", frame_stride=2, with_fvd=False, fvd_batch_size=4, fvd_max_items=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DISTS(load_weights=False).to(device)
    weights = torch.load("DISTS/DISTS_pytorch/weights.pt", map_location=device)
    model.alpha.data = weights["alpha"]
    model.beta.data = weights["beta"]
    model.eval()

    mse_vals = []
    dists_vals = []

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

            ref = torch.from_numpy(ref_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            dist = torch.from_numpy(dist_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            ref = ref.to(device)
            dist = dist.to(device)

            mse_vals.append(F.mse_loss(ref, dist).item())
            dists_vals.append(model(ref, dist).item())
            frame_idx += 1
    finally:
        ref_cap.release()
        dist_cap.release()

    frame_count = len(mse_vals)
    if frame_count == 0:
        raise ValueError("No frames read from one or both videos")

    result = {
        "mse_mean_over_t": sum(mse_vals) / frame_count,
        "dists_mean_over_t": sum(dists_vals) / frame_count,
        "frames": frame_count,
    }

    if with_fvd:
        fvd_path = Path(__file__).parent / "PyTorch-Frechet-Video-Distance"
        if not fvd_path.exists():
            raise FileNotFoundError("PyTorch-Frechet-Video-Distance submodule not found")
        sys.path.append(str(fvd_path))
        try:
            from fvd_metric import compute_fvd
        except Exception as exc:
            raise ImportError("Failed to import fvd_metric. Ensure dependencies are installed.") from exc

        ref_frames = load_video_frames(ref_path, frame_stride=frame_stride)
        if not ref_frames:
            raise ValueError("No frames read from ref video for FVD")
        ref_size = (ref_frames[0].shape[1], ref_frames[0].shape[0])
        dist_frames = load_video_frames(dist_path, frame_stride=frame_stride, target_size=ref_size)
        if not dist_frames:
            raise ValueError("No frames read from dist video for FVD")

        if len(ref_frames) != len(dist_frames):
            min_len = min(len(ref_frames), len(dist_frames))
            print(f"Warning: frame counts differ for FVD, truncating to {min_len} frames")
            ref_frames = ref_frames[:min_len]
            dist_frames = dist_frames[:min_len]

        ref_tensor = torch.from_numpy(np.stack(ref_frames)).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        dist_tensor = torch.from_numpy(np.stack(dist_frames)).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        ref_tensor = ref_tensor.to(device)
        dist_tensor = dist_tensor.to(device)

        result["fvd"] = compute_fvd(ref_tensor, dist_tensor, fvd_max_items, torch.device(device), fvd_batch_size)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="video/preview.mp4")
    parser.add_argument("--dist", default="video/preview_imagined.mp4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--with-fvd", action="store_true")
    parser.add_argument("--fvd-batch-size", type=int, default=4)
    parser.add_argument("--fvd-max-items", type=int, default=None)
    args = parser.parse_args()

    result = eval_videos(
        args.ref,
        args.dist,
        device=args.device,
        frame_stride=args.stride,
        with_fvd=args.with_fvd,
        fvd_batch_size=args.fvd_batch_size,
        fvd_max_items=args.fvd_max_items,
    )
    print(result)
