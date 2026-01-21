import cv2
import torch
import torch.nn.functional as F

from DISTS.DISTS_pytorch.DISTS_pt import DISTS


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def eval_videos(ref_path, dist_path, device=None, frame_stride=10):
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
                dist_frame = cv2.resize(
                    dist_frame,
                    (ref_frame.shape[1], ref_frame.shape[0]),
                    interpolation=cv2.INTER_AREA if dist_frame.shape[0] > ref_frame.shape[0] else cv2.INTER_LINEAR,
                )

            # [H,W,3] -> [1,3,H,W], 0-1
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

    return {
        "mse_mean_over_t": sum(mse_vals) / frame_count,
        "dists_mean_over_t": sum(dists_vals) / frame_count,
        "frames": frame_count,
    }


if __name__ == "__main__":
    result = eval_videos("video/preview.mp4", "video/preview_imagined.mp4")
    print(result)
