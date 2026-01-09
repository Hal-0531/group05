
import os
import glob
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 設定パラメータ
# ==========================================
DATA_DIR = "./datasets/movi_a/videos"
SAVE_DIR = "./checkpoints/savi_pixels_0108_128"
os.makedirs(SAVE_DIR, exist_ok=True)

# 画像設定 (Slot Attention標準の64x64)
IMG_SIZE = (64, 64) 

# 学習設定
BATCH_SIZE = 8      # メモリに乗るので大きめでOK
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
SEQ_LEN = 6         

# モデルハイパーパラメータ (軽量・高速設定)
NUM_SLOTS = 11        # まずは6個で確実に学習させる
SLOT_DIM = 128     
CNN_HIDDEN_DIM = 128
SLOT_ITERS = 5       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Dataset (爆速オンメモリ版)
# ==========================================
class MOViVideoDataset(Dataset):
    def __init__(self, video_dir, img_size=(64, 64), seq_len=6):
        self.img_size = img_size
        self.seq_len = seq_len
        self.videos = [] 

        files = sorted(glob.glob(os.path.join(video_dir, "*.avi")))
        print(f"[Init] Loading {len(files)} videos into RAM...")

        for filepath in tqdm(files):
            cap = cv2.VideoCapture(filepath)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # リサイズ (256 -> 64)
                if frame.shape[:2] != img_size:
                    frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
                
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if len(frames) > 0:
                self.videos.append(np.array(frames, dtype=np.uint8))
        
        print(f"[Init] Done. Loaded {len(self.videos)} videos.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames_np = self.videos[idx] 
        total_frames = len(frames_np)
        
        if total_frames < self.seq_len:
            padding = [frames_np[-1]] * (self.seq_len - total_frames)
            frames_np = np.concatenate([frames_np, np.array(padding)], axis=0)
            total_frames = len(frames_np)

        max_start = total_frames - self.seq_len
        start_idx = random.randint(0, max(0, max_start))
        
        clip_np = frames_np[start_idx : start_idx + self.seq_len]
        
        # float32 (0.0 - 1.0)
        clip_tensor = torch.tensor(clip_np.astype(np.float32) / 255.0)
        return clip_tensor.permute(0, 3, 1, 2)

# ==========================================
# 3. Model Components (修正版)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, hidden_dim, IMG_SIZE[0], IMG_SIZE[1]))
        nn.init.normal_(self.pos_emb, std=0.02) # 重みの初期化

    def forward(self, x):
        return self.net(x) + self.pos_emb

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim)
        )

    def forward(self, inputs, prev_slots=None):
        b, n, d = inputs.shape
        if prev_slots is None:
            mu = self.slots_mu.expand(b, self.num_slots, -1)
            sigma = self.slots_logsigma.expand(b, self.num_slots, -1).exp()
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = prev_slots

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, hidden_dim, resolution):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = nn.Parameter(torch.zeros(1, slot_dim, resolution[0], resolution[1]))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, 4, 3, padding=1) # Output: RGB(3) + Alpha(1)
        )

    def forward(self, slots):
        x = slots.reshape(-1, slots.shape[-1])
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.resolution[0], self.resolution[1])
        x = x + self.pos_emb
        out = self.net(x) # (N, 4, H, W)
        
        # ★★★ ここが最重要修正 ★★★
        # RGBには Sigmoid をかけて 0.0-1.0 に収める
        recons = torch.sigmoid(out[:, :3, :, :]) 
        # Alphaはそのまま (あとでSoftmaxするのでLogitsのままでいい)
        masks = out[:, 3:4, :, :] 
        
        return recons, masks

class SAVi_Pixels(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(CNN_HIDDEN_DIM)
        self.slot_attn = SlotAttention(NUM_SLOTS, SLOT_DIM, iters=SLOT_ITERS)
        self.decoder = SpatialBroadcastDecoder(SLOT_DIM, CNN_HIDDEN_DIM, IMG_SIZE)

    def forward(self, video):
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        feat = self.encoder(x)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(b, t, -1, CNN_HIDDEN_DIM)
        
        slots_t = None
        all_slots = []
        for step in range(t):
            feat_t = feat_flat[:, step]
            slots_t = self.slot_attn(feat_t, prev_slots=slots_t)
            all_slots.append(slots_t)
            
        slots_seq = torch.stack(all_slots, dim=1) # (B, T, S, D)
        
        # Decode
        slots_flat = slots_seq.reshape(b * t * NUM_SLOTS, SLOT_DIM)
        recons, masks = self.decoder(slots_flat) # reconsはSigmoid済み
        
        # Shape戻し
        recons = recons.reshape(b * t, NUM_SLOTS, 3, h, w)
        masks = masks.reshape(b * t, NUM_SLOTS, 1, h, w)
        
        # Mask Softmax
        masks = torch.softmax(masks, dim=1)
        
        # Combine
        recon_combined = torch.sum(recons * masks, dim=1)
        recon_combined = recon_combined.reshape(b, t, 3, h, w)
        
        # 戻り値用
        masks = masks.reshape(b, t, NUM_SLOTS, 1, h, w)
        recons = recons.reshape(b, t, NUM_SLOTS, 3, h, w)
        
        return recon_combined, masks, recons, slots_seq

# ==========================================
# 4. Training Loop
# ==========================================
def main():
    print("=== Start SAVi Pixel Training (Fixed) ===")
    
    dataset = MOViVideoDataset(DATA_DIR, img_size=IMG_SIZE, seq_len=SEQ_LEN)
    # Windowsでは num_workers=0 推奨
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    
    model = SAVi_Pixels().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, Slots: {NUM_SLOTS}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 混合精度学習 (AMP)
            with autocast():
                recon_combined, _, _, _ = model(batch)
                loss = criterion(recon_combined, batch)
            
            # バックプロパゲーション
            scaler.scale(loss).backward()

            # 勾配クリッピング (unscaleしてから行う)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)

            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(mse=f"{loss.item():.5f}")
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.5f}")
        
        # 保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"savi_pixel_ep{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, save_path)

if __name__ == "__main__":
    main()