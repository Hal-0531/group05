import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
class Config:
    # --- データパス ---
    DATA_FOLDER = r"./datasets/movi_a/videos" 
    
    # --- 保存設定 ---
    SAVE_DIR = "./checkpoints/pixel_transformer_dynamics"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    HISTORY_PLOT_PATH = os.path.join(SAVE_DIR, "training_history.png")
    IMAGE_PLOT_PREFIX = os.path.join(SAVE_DIR, "epoch_result")
    FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "pixel_transformer_final.pth")

    # --- 画像・データ仕様 ---
    IMG_SIZE = 64
    CHANNELS = 3
    FRAMES_PER_SAMPLE = 10
    
    # --- Transformer設定 (Slotモデルと統一) ---
    # 画像をパッチに分割してトークン化します
    PATCH_SIZE = 8  # 64x64 -> 8x8 grid = 64 tokens per frame
    
    D_MODEL = 128   # Slotモデルと同じ
    NHEAD = 4       # Slotモデルと同じ
    LAYERS = 4      # Slotモデルと同じ
    FF_DIM = 512    # Slotモデルと同じ
    DROPOUT = 0.1

    # --- 学習設定 ---
    BATCH_SIZE = 16 # メモリに応じて調整してください
    LEARNING_RATE = 5e-4 # Transformerは少し低めが安定します
    NUM_EPOCHS = 100
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. データセット (Raw Video)
# ==========================================
class VideoDataset(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.avi"))) # 拡張子は適宜調整
        # 拡張子フォールバック
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
            self.files = [f for f in self.files if f.endswith(('.mp4', '.mov', '.npy'))]
            
        self.img_size = Config.IMG_SIZE
        print(f"Found {len(self.files)} video files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frames = self._load_video(self.files[idx])
        # [T, H, W, C] -> [T, C, H, W] -> Normalize 0-1
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        
        T = Config.FRAMES_PER_SAMPLE
        if frames.shape[0] < T:
            padding = torch.zeros(T - frames.shape[0], *frames.shape[1:])
            frames = torch.cat([frames, padding], dim=0)
        else:
            frames = frames[:T]
        return frames

    def _load_video(self, path):
        if path.endswith('.npy'):
            return np.load(path)
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= Config.FRAMES_PER_SAMPLE: break
        cap.release()
        if len(frames) == 0: # エラー処理: 読み込めなかった場合
            return np.zeros((Config.FRAMES_PER_SAMPLE, self.img_size, self.img_size, 3), dtype=np.uint8)
        return np.array(frames)

# ==========================================
# 3. Model: Pixel Transformer (ViT-based Dynamics)
# ==========================================
class PixelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = Config.PATCH_SIZE
        self.d_model = Config.D_MODEL
        self.img_size = Config.IMG_SIZE
        
        # 1フレームあたりのパッチ数 (64 / 8)^2 = 64
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # --- Patch Embedding ---
        # Conv2dを使ってパッチ切り出しと埋め込みを同時に行う
        # Input: (B*T, 3, H, W) -> Output: (B*T, D, H/P, W/P)
        self.patch_embed = nn.Conv2d(
            in_channels=3, 
            out_channels=self.d_model, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # --- Positional Embeddings ---
        # 時間方向の埋め込み (10フレーム)
        self.pos_emb_time = nn.Parameter(torch.zeros(1, Config.FRAMES_PER_SAMPLE, 1, self.d_model))
        # 空間方向の埋め込み (64パッチ)
        self.pos_emb_spatial = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.NHEAD,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.LAYERS)
        
        # --- Output Head ---
        # パッチの特徴量からピクセルに戻す (D -> 3 * P * P)
        self.output_proj = nn.Linear(self.d_model, 3 * self.patch_size * self.patch_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        
        # 1. Patchify & Embedding
        # [B*T, C, H, W]
        # ★修正済み: .view() -> .reshape()
        x_flat = x.reshape(b * t, c, h, w)
        
        # [B*T, D, grid_h, grid_w]
        patches = self.patch_embed(x_flat)
        # [B*T, D, num_patches] -> [B*T, num_patches, D]
        patches = patches.flatten(2).transpose(1, 2)
        # [B, T, num_patches, D]
        # ★修正済み: .view() -> .reshape()
        patches = patches.reshape(b, t, self.num_patches, self.d_model)
        
        # 2. Add Positional Embeddings
        patches = patches + self.pos_emb_time[:, :t, :, :]
        patches = patches + self.pos_emb_spatial
        
        # 3. Flatten for Transformer
        # ★修正済み: .view() -> .reshape()
        x_seq = patches.reshape(b, t * self.num_patches, self.d_model)
        
        # 4. Causal Masking
        seq_len = x_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # 5. Transformer Forward
        out_seq = self.transformer(x_seq, mask=mask)
        
        # 6. Output Projection & Reshape back to Image
        out_pixels = self.output_proj(out_seq)
        
        # Shape戻し: [B, T, num_patches, 3*P*P]
        # ★修正済み: .view() -> .reshape()
        out_pixels = out_pixels.reshape(b, t, self.num_patches, 3 * self.patch_size * self.patch_size)
        
        # Patch結合 (Unpatchify)
        # ★修正済み: .view() -> .reshape()
        out_pixels = out_pixels.reshape(b, t, int(h/self.patch_size), int(w/self.patch_size), 3, self.patch_size, self.patch_size)
        
        # Permute to [B, T, 3, grid_h, patch_h, grid_w, patch_w]
        out_pixels = out_pixels.permute(0, 1, 4, 2, 5, 3, 6)
        
        # Combine axes to [B, T, 3, H, W]
        # ★修正済み: .view() -> .reshape()
        out_pixels = out_pixels.reshape(b, t, c, h, w)
        
        return torch.sigmoid(out_pixels)

# ==========================================
# 4. Utils (Metrics & Viz)
# ==========================================
def calculate_psnr(loss):
    if loss == 0: return 100.0
    return -10 * math.log10(loss)

def save_history_plot(history):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-o', label='MSE Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['psnr'], 'g-o', label='PSNR (dB)')
    plt.title('Reconstruction Quality')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Config.HISTORY_PLOT_PATH)
    plt.close()

def save_image_comparison(inputs, preds, epoch):
    # バッチの先頭のみ保存
    # inputs: [B, T, C, H, W]
    # ★修正箇所: .astype(np.float32) を追加して float16 エラーを回避
    inputs = inputs.detach().cpu().numpy()[0].astype(np.float32)
    preds = preds.detach().cpu().numpy()[0].astype(np.float32)
    
    T = inputs.shape[0]
    show_frames = min(T, 8)
    
    fig, axes = plt.subplots(2, show_frames, figsize=(show_frames*2, 4))
    if show_frames == 1: axes = np.array([axes]).T
    
    for t in range(show_frames):
        # GT
        ax = axes[0, t]
        img_gt = inputs[t].transpose(1, 2, 0) # CHW -> HWC
        ax.imshow(np.clip(img_gt, 0, 1))
        ax.axis('off')
        if t == 0: ax.set_title("GT")
        
        # Pred
        ax = axes[1, t]
        img_pred = preds[t].transpose(1, 2, 0)
        ax.imshow(np.clip(img_pred, 0, 1))
        ax.axis('off')
        if t == 0: ax.set_title("Pred")
        
    plt.tight_layout()
    plt.savefig(f"{Config.IMAGE_PLOT_PREFIX}_{epoch}.png")
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 5. Main Loop
# ==========================================
def main():
    print("=== Start Pixel Transformer Dynamics Training ===")
    
    dataset = VideoDataset(Config.DATA_FOLDER)
    # Windowsの場合は num_workers=0 を推奨
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    
    model = PixelTransformer().to(Config.DEVICE)
    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Structure: {Config.LAYERS} Layers, {Config.D_MODEL} Dim, {Config.NHEAD} Heads (Matched to Slot Model)")
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda') # for mixed precision (Updated for recent PyTorch)
    
    history = {'loss': [], 'psnr': []}
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        for batch in pbar:
            batch = batch.to(Config.DEVICE)
            
            # 入力: 0 ~ T-1 (過去)
            # 目標: 1 ~ T (未来) ※自己回帰学習
            inputs = batch[:, :-1] 
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # [B, T-1, C, H, W]
                preds = model(inputs)
                loss = criterion(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item():.5f}"})
        
        # Stats
        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = calculate_psnr(avg_loss)
        
        history['loss'].append(avg_loss)
        history['psnr'].append(avg_psnr)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB")
        
        # Save & Viz
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f"pixel_trans_ep{epoch+1}.pth"))
            save_image_comparison(targets, preds, epoch+1)
            save_history_plot(history)
            
    # Final Save
    torch.save(model.state_dict(), Config.FINAL_MODEL_PATH)
    print("Training Finished.")

if __name__ == "__main__":
    main()