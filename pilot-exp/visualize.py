import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
class Config:
    # --- ファイル指定 (ここを変更！) ---
    # 可視化したいファイルの番号またはIDを指定してください
    # 例: "00000", "00123", "video_02500" など
    TARGET_FILE_ID = "02500" 

    # --- パス設定 ---
    # スロットデータ(.pt)が入っているフォルダ
    SLOT_DIR = r"C:\Users\ibmkt\group5\datasets\movi_a\slots_savi_pixels"
    
    # 学習済みSAViのモデルファイル (.pth)
    CHECKPOINT_PATH = r"./checkpoints/savi_pixels/savi_pixel_ep60.pth"
    
    # --- モデル仕様 ---
    NUM_SLOTS = 11
    SLOT_DIM = 64
    IMG_SIZE = (64, 64)
    HIDDEN_DIM = 64
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Decoder定義 (学習時と同じ構造)
# ==========================================
class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, hidden_dim, resolution):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = nn.Parameter(torch.zeros(1, slot_dim, resolution[0], resolution[1]))
        
        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, 4, 3, padding=1) 
        )

    def forward(self, slots):
        x = slots.reshape(-1, slots.shape[-1])
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.resolution[0], self.resolution[1])
        x = x + self.pos_emb
        out = self.net(x)
        recons = torch.sigmoid(out[:, :3, :, :]) 
        masks = out[:, 3:4, :, :] 
        return recons, masks

# ==========================================
# 3. ロード & 可視化関数
# ==========================================
def load_decoder():
    print(f"Loading Decoder from {Config.CHECKPOINT_PATH}...")
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {Config.CHECKPOINT_PATH}")

    decoder = SpatialBroadcastDecoder(Config.SLOT_DIM, Config.HIDDEN_DIM, Config.IMG_SIZE).to(Config.DEVICE)
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    decoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            decoder_dict[k.replace('decoder.', '')] = v
    
    try:
        decoder.load_state_dict(decoder_dict, strict=True)
        print("Decoder weights loaded successfully (Strict).")
    except Exception as e:
        print(f"Warning: Loading with strict=False due to: {e}")
        decoder.load_state_dict(decoder_dict, strict=False)

    decoder.eval()
    return decoder

def visualize_reconstruction(decoder, slot_file_path):
    print(f"Processing: {os.path.basename(slot_file_path)}")
    
    try:
        slots = torch.load(slot_file_path, map_location=Config.DEVICE).float()
    except Exception as e:
        print(f"Error loading {slot_file_path}: {e}")
        return

    print(f"Slot Stats - Max: {slots.max().item():.3f}, Min: {slots.min().item():.3f}, Mean: {slots.abs().mean().item():.3f}")
    if slots.abs().max() < 1e-6:
        print("!!! Warning: Slots contain only Zeros! Output will be gray. !!!")

    num_frames = min(5, slots.shape[0])
    target_slots = slots[:num_frames]
    
    with torch.no_grad():
        T, S, D = target_slots.shape
        flat_slots = target_slots.reshape(T * S, D)
        recons, masks = decoder(flat_slots)
        
        recons = recons.reshape(T, S, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = masks.reshape(T, S, 1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = torch.softmax(masks, dim=1)
        combined = torch.sum(recons * masks, dim=1)
        
    combined = combined.permute(0, 2, 3, 1).cpu().numpy()
    masks_vis = masks.permute(0, 1, 3, 4, 2).cpu().numpy()

    # 画像表示
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2.5, 3))
    if num_frames == 1: axes = [axes]
    
    for i in range(num_frames):
        ax = axes[i]
        img = np.clip(combined[i], 0, 1)
        ax.imshow(img)
        ax.set_title(f"Frame {i}")
        ax.axis('off')
        
    plt.suptitle(f"Reconstruction: {os.path.basename(slot_file_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Slot Mask表示 (最初のフレームのみ)
    fig_m, axes_m = plt.subplots(1, S + 1, figsize=((S + 1) * 2, 2.5))
    axes_m[0].imshow(np.clip(combined[0], 0, 1))
    axes_m[0].set_title("Combined")
    axes_m[0].axis('off')
    for s in range(S):
        mask_img = masks_vis[0, s, :, :, 0]
        axes_m[s+1].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
        axes_m[s+1].set_title(f"Slot {s}")
        axes_m[s+1].axis('off')
    plt.suptitle("Slot Decomposition (Frame 0)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    # デコーダの準備
    decoder = load_decoder()
    
    # ファイル検索
    all_files = sorted(glob.glob(os.path.join(Config.SLOT_DIR, "*.pt")))
    if not all_files:
        print(f"No .pt files found in {Config.SLOT_DIR}")
        exit()

    # 指定されたIDを含むファイルを検索
    target_file = None
    for f in all_files:
        if Config.TARGET_FILE_ID in os.path.basename(f):
            target_file = f
            break
    
    if target_file:
        visualize_reconstruction(decoder, target_file)
    else:
        print(f"Error: File containing ID '{Config.TARGET_FILE_ID}' not found.")
        print("Available files (first 5):")
        for f in all_files[:5]:
            print(f" - {os.path.basename(f)}")