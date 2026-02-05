import os
# OpenMPエラー回避
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import glob
import random
import math

# ==========================================
# 1. 設定 (Fix: シンプルイズベスト)
# ==========================================
CONFIG = {
    "ROOT_DIR": "./1x/data/train_v2.0", 
    "SAVE_DIR": "./checkpoints/savi_capacity_fix",
    "CODEBOOK_PATH": "cosmos_codebook.pt", 
    
    "VOCAB_SIZE": 64000,
    "CODEBOOK_DIM": 6,
    "IMG_SIZE": (32, 32),
    
    "BATCH_SIZE": 8,          
    "ACCUMULATION_STEPS": 2,  
    "MAX_GRAD_NORM": 1.0,
    "NUM_EPOCHS": 50,         
    "LEARNING_RATE": 3e-4,    # 少し上げる
    "SEQ_LEN": 16,            
    
    # モデル設定 (Capacity Boost維持)
    "NUM_SLOTS": 6,           
    "SLOT_DIM": 256,          # 384次元
    "HIDDEN_DIM": 256,        
    "SLOT_ITERS": 3,
    
    "NUM_ENC_LAYERS": 4,      
    "NUM_HEADS": 4,           
    "DROPOUT": 0.0,
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Dataset
# ==========================================
class RandomClipDataset(Dataset):
    def __init__(self, config, num_samples_per_epoch=2000):
        self.root = config["ROOT_DIR"]
        self.seq_len = config["SEQ_LEN"]
        self.num_samples = num_samples_per_epoch 
        self.img_size = config["IMG_SIZE"]
        self.video_paths = sorted(glob.glob(os.path.join(self.root, "videos", "video_*.bin")))
        if len(self.video_paths) == 0: raise FileNotFoundError("No video files found.")

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        vid_path = random.choice(self.video_paths)
        vid_size = os.path.getsize(vid_path)
        bytes_per_frame = 2 * self.img_size[0] * self.img_size[1]
        total_frames = vid_size // bytes_per_frame
        if total_frames <= self.seq_len: start_frame = 0
        else: start_frame = random.randint(0, total_frames - self.seq_len)
        clip = np.memmap(
            vid_path, dtype=np.uint16, mode='r', 
            offset=start_frame * bytes_per_frame, 
            shape=(self.seq_len, self.img_size[0], self.img_size[1])
        ).copy() 
        return torch.from_numpy(clip.astype(np.int64))

# ==========================================
# 3. Model Components
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CodebookEncoder(nn.Module):
    def __init__(self, codebook_path, hidden_dim, num_layers, nhead, dropout, resolution):
        super().__init__()
        codebook_weight = torch.load(codebook_path).float()
        self.code_emb = nn.Embedding.from_pretrained(codebook_weight, freeze=True)
        self.input_proj = nn.Linear(codebook_weight.shape[1], hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=resolution[0]*resolution[1])
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_tokens):
        b, h, w = x_tokens.shape
        x_flat = x_tokens.view(b, h*w)
        x_vec = self.code_emb(x_flat) 
        x_feat = self.input_proj(x_vec)
        x_feat = self.pos_enc(x_feat)
        features = self.transformer_encoder(x_feat)
        return self.norm(features), x_vec

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=384):
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
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, dim))

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
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn_norm)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, output_dim, resolution):
        super().__init__()
        self.resolution = resolution
        self.pos_emb = nn.Parameter(torch.randn(1, slot_dim, resolution[0], resolution[1]) * 0.02)
        
        # CNN Backbone (Capacity Boost)
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(slot_dim, slot_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(slot_dim, slot_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(slot_dim, slot_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(slot_dim, slot_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(slot_dim, output_dim + 1, 1) 
        )

    def forward(self, slots):
        b, num_slots, d = slots.shape
        h, w = self.resolution
        slots_flat = slots.view(b * num_slots, d, 1, 1).expand(-1, -1, h, w)
        pos_emb = self.pos_emb.expand(b * num_slots, -1, -1, -1)
        x = slots_flat + pos_emb
        out = self.decoder_cnn(x) 
        
        vecs, alpha = torch.split(out, [6, 1], dim=1)
        
        # ★ Tanh撤廃: Linearに戻す
        # vecs = torch.tanh(vecs) 
        
        vecs = vecs.view(b, num_slots, 6, h, w)
        alpha = alpha.view(b, num_slots, 1, h, w)
        mask = torch.softmax(alpha, dim=1)
        recon_vecs = torch.sum(vecs * mask, dim=1) 
        return recon_vecs.permute(0, 2, 3, 1) 

class CapacitySAVi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CodebookEncoder(config["CODEBOOK_PATH"], config["HIDDEN_DIM"], config["NUM_ENC_LAYERS"], config["NUM_HEADS"], config["DROPOUT"], config["IMG_SIZE"])
        self.slot_attn = SlotAttention(config["NUM_SLOTS"], config["SLOT_DIM"], config["SLOT_ITERS"], config["HIDDEN_DIM"])
        self.decoder = SpatialBroadcastDecoder(config["SLOT_DIM"], config["CODEBOOK_DIM"], config["IMG_SIZE"])

    def forward(self, video_tokens):
        b, t, h, w = video_tokens.shape
        slots_t = None
        all_recon = []
        all_target = []
        for step in range(t):
            frame = video_tokens[:, step]
            feat, target = self.encoder(frame) 
            slots_t = self.slot_attn(feat, prev_slots=slots_t)
            recon = self.decoder(slots_t)
            recon = recon.view(b, h*w, 6)
            all_recon.append(recon)
            all_target.append(target)
        return torch.stack(all_recon, dim=1), torch.stack(all_target, dim=1)

# ==========================================
# 4. Main Training Loop (Standard MSE)
# ==========================================
def main():
    print("=== Start Capacity Fix Training (No Tanh, No Hard Mining) ===")
    
    dataset = RandomClipDataset(CONFIG, num_samples_per_epoch=2000)
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=0)
    
    model = CapacitySAVi(CONFIG).to(DEVICE)
    print("Initializing large model from scratch...")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-4)
    mse_crit = nn.MSELoss()
    acc_steps = CONFIG["ACCUMULATION_STEPS"]
    
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            recon, target = model(batch)
            
            # 普通のMSEに戻す
            loss = mse_crit(recon, target)
            
            loss_accum = loss / acc_steps
            loss_accum.backward()
            total_loss += loss.item()
            
            if (batch_idx + 1) % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["MAX_GRAD_NORM"])
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(mse=f"{loss.item():.5f}")
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg MSE: {avg_loss:.5f}")
        
        torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "savi_capacity_fix_latest.pth"))

if __name__ == "__main__":
    main()