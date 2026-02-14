import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


FRAME_RE = re.compile(r"^(gt|pred)_(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mp4 videos from frame folders (video_xxxxx/gt_###.png, pred_###.png)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "slot_dynamics/results_frames",
            "pixel_dynamics/results_pixel_frames",
        ],
        help="One or more dataset roots that contain video_* folders.",
    )
    parser.add_argument(
        "--output-root",
        default="generated_videos",
        help="Directory where generated videos will be written.",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="FPS for output mp4 files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def collect_frames(video_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = {"gt": [], "pred": []}
    for p in video_dir.glob("*.png"):
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        frame_type = m.group(1).lower()
        frame_idx = int(m.group(2))
        groups[frame_type].append((frame_idx, p))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def validate_pairs(gt_frames: List[Tuple[int, Path]], pred_frames: List[Tuple[int, Path]], video_dir: Path) -> None:
    if not gt_frames or not pred_frames:
        raise ValueError(f"{video_dir}: gt/pred frames are missing.")
    gt_idx = [idx for idx, _ in gt_frames]
    pred_idx = [idx for idx, _ in pred_frames]
    if gt_idx != pred_idx:
        raise ValueError(f"{video_dir}: frame indices mismatch (gt={gt_idx}, pred={pred_idx}).")


def write_video(frame_items: List[Tuple[int, Path]], out_path: Path, fps: float, overwrite: bool) -> int:
    if out_path.exists() and not overwrite:
        return 0

    first_img = cv2.imread(str(frame_items[0][1]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise ValueError(f"Failed to read image: {frame_items[0][1]}")
    height, width = first_img.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for: {out_path}")

    try:
        written = 0
        for _, frame_path in frame_items:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read image: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written


def process_dataset(input_root: Path, output_root: Path, fps: float, overwrite: bool) -> None:
    if not input_root.exists():
        print(f"[WARN] Skip missing input: {input_root}")
        return

    video_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("video_")])
    if not video_dirs:
        print(f"[WARN] No video_* dirs found in: {input_root}")
        return

    dataset_name = input_root.parent.name
    dataset_output = output_root / dataset_name
    converted = 0
    skipped = 0

    print(f"[INFO] Dataset: {dataset_name} ({len(video_dirs)} folders)")
    for video_dir in video_dirs:
        frames = collect_frames(video_dir)
        validate_pairs(frames["gt"], frames["pred"], video_dir)

        out_dir = dataset_output / video_dir.name
        gt_out = out_dir / "gt.mp4"
        pred_out = out_dir / "pred.mp4"

        gt_written = write_video(frames["gt"], gt_out, fps, overwrite)
        pred_written = write_video(frames["pred"], pred_out, fps, overwrite)

        if gt_written == 0 and pred_written == 0:
            skipped += 1
        else:
            converted += 1

    print(f"[INFO] {dataset_name}: converted={converted}, skipped={skipped}, output={dataset_output}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for input_dir in args.inputs:
        process_dataset(Path(input_dir), output_root, args.fps, args.overwrite)


if __name__ == "__main__":
    main()
