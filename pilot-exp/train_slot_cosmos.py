import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler # 新しいインポート方法に対応
import numpy as np
from tqdm import tqdm

# ==========================================
# 0. Windows環境設定 (メモリ不足対策)
# ==========================================
# GPUメモリの断片化を防ぐ設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 設定パラメータ
# ==========================================
DATA_DIR = "./datasets/movi_a/tokens_dv128"
SAVE_DIR = "./checkpoints/savi_cosmos_0109"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cosmos Tokenizerの仕様
INPUT_RES = (32, 32)
VOCAB_SIZE = 65536

# ★修正: メモリ対策設定
BATCH_SIZE = 2        # 8 -> 2 に削減 (16GB VRAM向け)
GRAD_ACCUM_STEPS = 4  # 4回分勾配を貯める (実質バッチサイズ = 2 * 4 = 8)

NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
SEQ_LEN = 4

# モデルハイパーパラメータ
NUM_SLOTS = 11
SLOT_DIM = 32
HIDDEN_DIM = 32
SLOT_ITERS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Dataset
# ==========================================
class TokenVideoDataset(Dataset):
    def __init__(self, token_dir, seq_len=3):
        self.seq_len = seq_len
        self.files = sorted(glob.glob(os.path.join(token_dir, "*.pt")))
        print(f"[Init] Found {len(self.files)} token files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            token_tensor = torch.load(self.files[idx], weights_only=True) # セキュリティ警告回避推奨
        except:
            token_tensor = torch.load(self.files[idx])

        if token_tensor.dim() == 4:
            token_tensor = token_tensor.squeeze(0)
            
        token_tensor = token_tensor.long()
        total_frames = token_tensor.shape[0]
        
        if total_frames < self.seq_len:
            padding = token_tensor[-1:].repeat(self.seq_len - total_frames, 1, 1)
            token_tensor = torch.cat([token_tensor, padding], dim=0)
        elif total_frames > self.seq_len:
            max_start = total_frames - self.seq_len
            start = random.randint(0, max_start)
            token_tensor = token_tensor[start : start + self.seq_len]

        return token_tensor

# ==========================================
# 3. Model Components
# ==========================================
class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, resolution):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, hidden_dim, resolution[0], resolution[1]))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        # x: (B*T, H, W)
        x = self.embedding(x) # (B*T, H, W, D)
        x = x.permute(0, 3, 1, 2) # (B*T, D, H, W)
        feat = self.net(x)
        return feat + self.pos_emb

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

class TokenBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, hidden_dim, vocab_size, resolution):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = nn.Parameter(torch.zeros(1, slot_dim, resolution[0], resolution[1]))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim + 1, 1)
        )
        self.final_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, slots):
        x = slots.reshape(-1, slots.shape[-1])
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.resolution[0], self.resolution[1])
        x = x + self.pos_emb
        out = self.net(x)
        feat = out[:, :-1, :, :]
        mask_logits = out[:, -1:, :, :]
        return feat, mask_logits

class SAVi_Tokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TokenEncoder(VOCAB_SIZE, HIDDEN_DIM, INPUT_RES)
        self.slot_attn = SlotAttention(NUM_SLOTS, SLOT_DIM, iters=SLOT_ITERS)
        self.decoder = TokenBroadcastDecoder(SLOT_DIM, HIDDEN_DIM, VOCAB_SIZE, INPUT_RES)

    def forward(self, token_indices):
        b, t, h, w = token_indices.shape
        x_flat = token_indices.reshape(b * t, h, w)
        feat = self.encoder(x_flat)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(b, t, -1, HIDDEN_DIM)
        
        slots_t = None
        all_slots = []
        for step in range(t):
            feat_t = feat_flat[:, step]
            slots_t = self.slot_attn(feat_t, prev_slots=slots_t)
            all_slots.append(slots_t)
            
        slots_seq = torch.stack(all_slots, dim=1)
        slots_flat = slots_seq.reshape(b * t * NUM_SLOTS, SLOT_DIM)
        slot_feats, mask_logits = self.decoder(slots_flat)
        
        slot_feats = slot_feats.reshape(b * t, NUM_SLOTS, HIDDEN_DIM, h, w)
        mask_logits = mask_logits.reshape(b * t, NUM_SLOTS, 1, h, w)
        masks = torch.softmax(mask_logits, dim=1)
        
        feat_combined = torch.sum(slot_feats * masks, dim=1)
        feat_combined_perm = feat_combined.permute(0, 2, 3, 1) # (B*T, H, W, D)
        
        # メモリ節約のため、ここでShapeを戻さず、FlatなままLinearに通す
        logits = self.decoder.final_proj(feat_combined_perm) # (B*T, H, W, Vocab)
        
        # 戻り値を整形 (B, T, H, W, Vocab)
        logits = logits.reshape(b, t, h, w, VOCAB_SIZE)
        masks = masks.reshape(b, t, NUM_SLOTS, 1, h, w)
        
        return logits, masks

# ==========================================
# 4. Training Loop (勾配蓄積版)
# ==========================================
def main():
    print("=== Start SAVi Token Training (Optimized) ===")
    print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"Vocab Size: {VOCAB_SIZE}, Input Res: {INPUT_RES}")
    
    dataset = TokenVideoDataset(DATA_DIR, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    
    model = SAVi_Tokens().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda') # Argument fix
    
    print(f"Device: {DEVICE}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Epoch開始時に初期化
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            
            # --- Forward ---
            with autocast(device_type='cuda'):
                logits, _ = model(batch)
                
                # Reshape for Loss
                logits_flat = logits.reshape(-1, VOCAB_SIZE)
                target_flat = batch.reshape(-1)
                
                loss = criterion(logits_flat, target_flat)
                
                # 勾配蓄積のためにLossを割る
                loss = loss / GRAD_ACCUM_STEPS
            
            # --- Backward ---
            scaler.scale(loss).backward()
            
            # --- Step (一定回数ごとに更新) ---
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 更新後に初期化
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS # 表示用に戻す
            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"savi_token_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()