import os
# OpenMPエラー回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import av
from torch.utils.data import Dataset
from tqdm import tqdm

# ★ 学習コードからモデルクラスと設定を読み込み
# ※ train_capacity_boost.py が同じフォルダにある必要があります
try:
    from sanity_slot import CapacitySAVi, CONFIG
except ImportError:
    print("Error: 'train_capacity_boost.py' が見つかりません。同じフォルダに配置してください。")
    exit()

# ==========================================
# 設定
# ==========================================
VIS_SETTINGS = {
    # 学習済みモデルのパス (最新のエポックを指定)
    "CHECKPOINT_PATH": "./checkpoints/savi_capacity_boost/savi_capacity_epoch_10.pth",
    
    # 必要なファイル
    "CODEBOOK_PATH": "cosmos_codebook.pt",
    "DECODER_JIT_PATH": r"C:\Users\ibmkt\group5\models\Cosmos-Tokenizer-DV8x8x8\decoder.jit",
    "ROOT_DIR": r"C:\Users\ibmkt\group5\Fase2\1x\data\train_v2.0",
    
    # 出力設定
    "OUTPUT_DIR": "./capacity_results",
    "NUM_SAMPLES": 3,   # 最初の3つの動画を処理
    "SEQ_LEN": 30,      # 30フレーム分生成
    "FPS": 30,
    "IMG_SIZE": (32, 32)
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. データセット (動画指定読み込み)
# ==========================================
class SingleVideoDataset(Dataset):
    def __init__(self, root_dir, shard_id, seq_len=30, img_size=(32, 32)):
        self.vid_path = os.path.join(root_dir, "videos", f"video_{shard_id}.bin")
        if not os.path.exists(self.vid_path):
            raise FileNotFoundError(f"Video not found: {self.vid_path}")
            
        self.img_size = img_size
        self.seq_len = seq_len
        self.bytes_per_frame = 2 * img_size[0] * img_size[1]
        
        self.video_mem = np.memmap(
            self.vid_path, dtype=np.uint16, mode='r', 
            shape=(os.path.getsize(self.vid_path)//self.bytes_per_frame, img_size[0], img_size[1])
        )

    def __len__(self): return 1

    def __getitem__(self, idx):
        clip = self.video_mem[:self.seq_len].copy()
        return torch.from_numpy(clip.astype(np.int64))

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def decode_tokens_to_video(tokens, decoder_jit):
    """Token IDs -> RGB Frames (using JIT)"""
    tokens = tokens.long().to(DEVICE)
    if tokens.ndim == 3: tokens = tokens.unsqueeze(0)
    
    with torch.no_grad():
        video_bf16 = decoder_jit(tokens)
        
    video_f32 = video_bf16.squeeze(0).float()
    video_norm = (video_f32.clamp(-1, 1) + 1) / 2.0
    video_np = video_norm.permute(1, 2, 3, 0).cpu().numpy()
    return (video_np * 255).astype(np.uint8)

def save_comparison_video(gt_frames, pred_frames, path, fps=30):
    """左:GT, 右:Pred の動画を保存"""
    T, H, W, C = gt_frames.shape
    border = 4
    
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W * 2 + border
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {'crf': '20'}
    
    print(f"  Saving video to {path}...")
    
    for t in range(T):
        frame_l = gt_frames[t]
        frame_r = pred_frames[t]
        sep = np.ones((H, border, C), dtype=np.uint8) * 255
        combined = np.concatenate([frame_l, sep, frame_r], axis=1)
        
        packet = av.VideoFrame.from_ndarray(combined, format="rgb24")
        for p in stream.encode(packet): container.mux(p)
            
    for p in stream.encode(): container.mux(p)
    container.close()

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("=== Start Capacity Boost Model Visualization ===")
    os.makedirs(VIS_SETTINGS["OUTPUT_DIR"], exist_ok=True)
    
    # 1. Load Resources
    print("Loading Codebook...")
    codebook = torch.load(VIS_SETTINGS["CODEBOOK_PATH"]).to(DEVICE).float()
    
    print("Loading JIT Decoder...")
    decoder_jit = torch.jit.load(VIS_SETTINGS["DECODER_JIT_PATH"]).to(DEVICE).eval()
    
    print(f"Loading Model from {VIS_SETTINGS['CHECKPOINT_PATH']}...")
    model = CapacitySAVi(CONFIG).to(DEVICE)
    if os.path.exists(VIS_SETTINGS["CHECKPOINT_PATH"]):
        state = torch.load(VIS_SETTINGS["CHECKPOINT_PATH"], map_location=DEVICE)
        model.load_state_dict(state)
        print("Model weights loaded.")
    else:
        print("Error: Checkpoint not found.")
        return
    model.eval()
    
    # 2. Process Videos
    for i in range(VIS_SETTINGS["NUM_SAMPLES"]):
        print(f"\nProcessing Video {i}...")
        try:
            ds = SingleVideoDataset(VIS_SETTINGS["ROOT_DIR"], shard_id=i, seq_len=VIS_SETTINGS["SEQ_LEN"])
            gt_tokens = ds[0].unsqueeze(0).to(DEVICE) # [1, T, 32, 32]
            
            # --- Inference (Get Vectors) ---
            with torch.no_grad():
                # recon_vecs: [1, T, L, 6] (Flattened spatial dims)
                recon_vecs, _ = model(gt_tokens)
                
            # --- Vector -> Token ID (Nearest Neighbor) ---
            print("  Finding nearest neighbors (Vector -> TokenID)...")
            B, T, L, D = recon_vecs.shape
            H, W = VIS_SETTINGS["IMG_SIZE"]
            
            # Reshape for calculation
            flat_pred = recon_vecs.reshape(-1, D) # [N, 6]
            
            # Chunk processing to save memory
            chunk_size = 10000
            pred_tokens_list = []
            
            for start in range(0, flat_pred.shape[0], chunk_size):
                end = min(start + chunk_size, flat_pred.shape[0])
                chunk = flat_pred[start:end]
                
                # Distance calculation (Euclidean)
                dists = torch.cdist(chunk, codebook)
                nearest_ids = dists.argmin(dim=1)
                pred_tokens_list.append(nearest_ids)
                
            # Reconstruct shape
            pred_tokens = torch.cat(pred_tokens_list).view(B, T, H, W)
            
            # --- Metrics ---
            acc = (pred_tokens == gt_tokens).float().mean().item() * 100
            print(f"  Token Accuracy: {acc:.2f}%")
            
            # --- Visualization ---
            print("  Decoding to pixels...")
            gt_video = decode_tokens_to_video(gt_tokens, decoder_jit)
            pred_video = decode_tokens_to_video(pred_tokens, decoder_jit)
            
            out_name = f"capacity_video_{i}_acc{acc:.1f}.mp4"
            save_comparison_video(
                gt_video, pred_video, 
                os.path.join(VIS_SETTINGS["OUTPUT_DIR"], out_name)
            )
            print("  Done.")
            
        except FileNotFoundError:
            print(f"Video {i} not found. Skipping.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()