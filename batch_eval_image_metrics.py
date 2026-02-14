import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval_image_metrics_torchmetrics import (
    DEFAULT_METRIC_NAMES,
    DEFAULT_SKIP_FIRST_N,
    eval_video_metrics_torchmetrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate generated gt/pred mp4 pairs with torchmetrics."
    )
    parser.add_argument(
        "--root",
        default="generated_videos",
        help="Root directory that contains dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["slot_dynamics", "pixel_dynamics"],
        help="Dataset folder names under --root.",
    )
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for evaluation.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRIC_NAMES),
        help="Comma-separated metrics. e.g. psnr,ssim,ms_ssim,lpips,vif,uiqi,sam,sdi",
    )
    parser.add_argument(
        "--output-json",
        default="result/batch_eval_results.json",
        help="Path to write summary + per-video results.",
    )
    parser.add_argument(
        "--output-csv",
        default="result/batch_eval_results.csv",
        help="Path to write per-video tabular results.",
    )
    return parser.parse_args()


def summarize_values(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    summary: Dict[str, float] = {
        "count": int(arr.size),
        "inf_count": int(np.isinf(arr).sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "mean": float(np.mean(arr)) if arr.size > 0 else None,
        "median": float(np.median(arr)) if arr.size > 0 else None,
        "std": float(np.std(arr)) if arr.size > 0 else None,
        "finite_mean": float(np.mean(finite)) if finite.size > 0 else None,
        "finite_median": float(np.median(finite)) if finite.size > 0 else None,
    }
    return summary


def evaluate_dataset(
    dataset_root: Path,
    dataset_name: str,
    metric_names: List[str],
    device: str,
    stride: int,
) -> List[Dict[str, float]]:
    if not dataset_root.exists():
        print(f"[WARN] dataset missing: {dataset_root}")
        return []

    video_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("video_")])
    rows: List[Dict[str, float]] = []
    print(f"[INFO] {dataset_name}: evaluating {len(video_dirs)} videos")

    for i, video_dir in enumerate(video_dirs, start=1):
        gt = video_dir / "gt.mp4"
        pred = video_dir / "pred.mp4"
        if not gt.exists() or not pred.exists():
            print(f"[WARN] {dataset_name}/{video_dir.name}: missing gt.mp4 or pred.mp4")
            continue

        result = eval_video_metrics_torchmetrics(
            str(gt),
            str(pred),
            device=device,
            frame_stride=stride,
            metric_names=metric_names,
        )

        row: Dict[str, float] = {
            "dataset": dataset_name,
            "video": video_dir.name,
            "frames": result["frames"],
        }
        for key, value in result.items():
            if key == "frames":
                continue
            row[key] = value
        rows.append(row)
        print(f"[INFO] {dataset_name}: {i}/{len(video_dirs)} {video_dir.name} done")
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    all_rows: List[Dict[str, float]] = []
    for dataset in args.datasets:
        dataset_root = root / dataset
        rows = evaluate_dataset(
            dataset_root=dataset_root,
            dataset_name=dataset,
            metric_names=metric_names,
            device=args.device,
            stride=args.stride,
        )
        all_rows.extend(rows)

    metric_result_keys = [f"{m}_mean_over_t" for m in metric_names]
    summary: Dict[str, Dict] = {}
    for dataset in args.datasets:
        ds_rows = [r for r in all_rows if r["dataset"] == dataset]
        ds_summary: Dict[str, Dict] = {
            "videos": len(ds_rows),
            "frames_total": int(sum(int(r["frames"]) for r in ds_rows)),
            "metrics": {},
        }
        for key in metric_result_keys:
            vals = [float(r[key]) for r in ds_rows if key in r]
            ds_summary["metrics"][key] = summarize_values(vals)
        summary[dataset] = ds_summary

    output = {
        "config": {
            "root": str(root),
            "datasets": args.datasets,
            "device": args.device,
            "stride": args.stride,
            "skip_first_n_fixed": DEFAULT_SKIP_FIRST_N,
            "metrics": metric_names,
        },
        "summary": summary,
        "results": all_rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[INFO] wrote JSON: {out_json}")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "video", "frames"] + metric_result_keys
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[INFO] wrote CSV: {out_csv}")

    print("[INFO] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
