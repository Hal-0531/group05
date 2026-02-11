import os
import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # common keys
        for k in ("tokens", "token", "data", "arr", "array", "x"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Unsupported type in {path}: {type(obj)}")
    return obj


def infer_mode(t: torch.Tensor) -> str:
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return "discrete"
    return "continuous"


def flatten_pair(gt: torch.Tensor, pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)}")
    return gt.reshape(-1), pred.reshape(-1)


def mse(x, y):
    return torch.mean((x - y) ** 2).item()


def mae(x, y):
    return torch.mean(torch.abs(x - y)).item()


def psnr_from_mse(m):
    if m == 0:
        return float("inf")
    return -10.0 * np.log10(m)


def token_accuracy(gt, pred):
    return torch.mean((gt == pred).float()).item()


def topk_accuracy(gt: torch.Tensor, logits: torch.Tensor, k: int) -> float:
    # logits: [N, V]
    topk = torch.topk(logits, k, dim=-1).indices
    return torch.mean((topk == gt.unsqueeze(-1)).any(dim=-1).float()).item()


def main():
    ap = argparse.ArgumentParser(description="Evaluate token prediction from .pt files.")
    ap.add_argument("--gt", required=True, help="Path to ground-truth .pt")
    ap.add_argument("--pred", required=True, help="Path to predicted .pt")
    ap.add_argument("--mode", choices=["auto", "continuous", "discrete"], default="auto")
    ap.add_argument("--save-json", default=None, help="Optional path to save metrics JSON")
    args = ap.parse_args()

    gt = load_tensor(args.gt)
    pred = load_tensor(args.pred)

    if args.mode == "auto":
        mode = infer_mode(gt)
    else:
        mode = args.mode

    gt_flat, pred_flat = flatten_pair(gt, pred)

    metrics = {}
    if mode == "continuous":
        m = mse(gt_flat.float(), pred_flat.float())
        metrics["mse"] = m
        metrics["mae"] = mae(gt_flat.float(), pred_flat.float())
        metrics["psnr"] = psnr_from_mse(m)
    else:
        # discrete
        metrics["token_accuracy"] = token_accuracy(gt_flat.long(), pred_flat.long())

    print(json.dumps({"mode": mode, **metrics}, indent=2))

    if args.save_json:
        Path(args.save_json).write_text(json.dumps({"mode": mode, **metrics}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
