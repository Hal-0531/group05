import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 設定パラメータ
# ==========================================
class Config:
    # --- パス設定 ---
    # 正常動作が確認できたSlotデータのフォルダ
    SLOT_DATA_DIR = r"C:\Users\ibmkt\group5\datasets\movi_a\slots_savi_pixels" 
    
    # 正常動作が確認できたSAViのチェックポイント
    SAVI_CHECKPOINT = r"./checkpoints/savi_pixels/savi_pixel_ep60.pth"
    
    SAVE_DIR = "./checkpoints/savi_latent_dynamics"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 保存ファイル名
    HISTORY_PLOT_PATH = os.path.join(SAVE_DIR, "training_history.png")
    IMAGE_PLOT_PREFIX = os.path.join(SAVE_DIR, "epoch_result")
    SANITY_CHECK_PATH = os.path.join(SAVE_DIR, "sanity_check_init.png")

    # --- データ仕様 ---
    SEQ_LEN = 10         
    NUM_SLOTS = 11       
    SLOT_DIM = 64        
    IMG_SIZE = (64, 64)  
    HIDDEN_DIM = 64 # Decoder中間層

    # --- Dynamicsモデル設定 ---
    DYN_D_MODEL = 128
    DYN_NHEAD = 4
    DYN_LAYERS = 4
    DYN_FF_DIM = 512
    
    # 学習設定
    BATCH_SIZE = 64      
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Dataset (.pt Loader)
# ==========================================
class SlotPtDataset(Dataset):
    def __init__(self, folder_path, seq_len=10):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
        self.seq_len = seq_len
        print(f"Found {len(self.files)} slot files (.pt) in {folder_path}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # map_location='cpu' でGPUメモリ圧迫を防ぐ
        slots = torch.load(self.files[idx], map_location='cpu') 
        slots = slots.detach().float()
        
        total_frames = slots.shape[0]
        
        if total_frames < self.seq_len:
            pad_len = self.seq_len - total_frames
            padding = slots[-1:].repeat(pad_len, 1, 1)
            slots = torch.cat([slots, padding], dim=0)
        
        if total_frames > self.seq_len:
            start = torch.randint(0, total_frames - self.seq_len, (1,)).item()
            slots = slots[start : start + self.seq_len]
            
        return slots

# ==========================================
# 3. Models
# ==========================================
class SlotDynamicsTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(Config.SLOT_DIM, Config.DYN_D_MODEL)
        
        self.pos_emb_time = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, 1, Config.DYN_D_MODEL))
        self.pos_emb_slot = nn.Parameter(torch.zeros(1, 1, Config.NUM_SLOTS, Config.DYN_D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.DYN_D_MODEL,
            nhead=Config.DYN_NHEAD,
            dim_feedforward=Config.DYN_FF_DIM,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.DYN_LAYERS)
        self.output_proj = nn.Linear(Config.DYN_D_MODEL, Config.SLOT_DIM)

    def forward(self, x):
        B, T, S, D = x.shape
        h = self.input_proj(x)
        h = h + self.pos_emb_time[:, :T, :, :] 
        h = h + self.pos_emb_slot
        
        h_flat = h.reshape(B, T * S, -1)
        # Causal Mask
        mask = torch.triu(torch.ones(T * S, T * S) * float('-inf'), diagonal=1).to(x.device)
        h_flat = self.transformer(h_flat, mask=mask)
        
        out = self.output_proj(h_flat)
        return out.reshape(B, T, S, -1)

# --- Decoder (Validation用) ---
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

def load_frozen_decoder():
    print(f"Loading Decoder from {Config.SAVI_CHECKPOINT}...")
    if not os.path.exists(Config.SAVI_CHECKPOINT):
        print("!!! ERROR: Checkpoint not found. Visualization will be broken. !!!")
        return SpatialBroadcastDecoder(Config.SLOT_DIM, Config.HIDDEN_DIM, Config.IMG_SIZE).to(Config.DEVICE)

    decoder = SpatialBroadcastDecoder(Config.SLOT_DIM, Config.HIDDEN_DIM, Config.IMG_SIZE).to(Config.DEVICE)
    checkpoint = torch.load(Config.SAVI_CHECKPOINT, map_location=Config.DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    decoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            decoder_dict[k.replace('decoder.', '')] = v
            
    print(f"Decoder keys found: {len(decoder_dict)}")
    
    try:
        decoder.load_state_dict(decoder_dict, strict=True)
        print("Decoder loaded successfully (Strict Mode).")
    except Exception as e:
        print(f"Warning: Loading with strict=False. Reason: {e}")
        decoder.load_state_dict(decoder_dict, strict=False)

    decoder.eval()
    for p in decoder.parameters(): p.requires_grad = False
    return decoder

# ==========================================
# 4. 可視化・ヘルパー関数
# ==========================================
def calculate_psnr(loss):
    if loss == 0: return 100.0
    return -10 * math.log10(loss)

def save_history_plot(history):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-o', label='MSE')
    plt.title('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['psnr'], 'g-o', label='PSNR')
    plt.title('Latent PSNR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Config.HISTORY_PLOT_PATH)
    plt.close()

def decode_slots_to_image(decoder, slots):
    # slots: [T, S, D]
    T, S, D = slots.shape
    with torch.no_grad():
        flat = slots.reshape(T * S, D)
        recons, masks = decoder(flat)
        recons = recons.reshape(T, S, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = masks.reshape(T, S, 1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = torch.softmax(masks, dim=1)
        combined = torch.sum(recons * masks, dim=1)
        combined = combined.permute(0, 2, 3, 1)
    return combined.detach().cpu().numpy()

def save_image_comparison(decoder, target_slots, pred_slots, epoch, save_path=None):
    tgt = target_slots[0]   # [T, S, D]
    pred = pred_slots[0]    # [T, S, D]
    
    tgt_imgs = decode_slots_to_image(decoder, tgt)
    pred_imgs = decode_slots_to_image(decoder, pred)
    
    T = tgt.shape[0]
    show_frames = min(T, 5)
    
    fig, axes = plt.subplots(2, show_frames, figsize=(show_frames*2, 4))
    if show_frames == 1: axes = np.array([axes]).T

    for t in range(show_frames):
        ax = axes[0, t]
        ax.imshow(np.clip(tgt_imgs[t], 0, 1))
        ax.axis('off')
        if t == 0: ax.set_title("GT")
        
        ax = axes[1, t]
        ax.imshow(np.clip(pred_imgs[t], 0, 1))
        ax.axis('off')
        if t == 0: ax.set_title("Pred")
        
    plt.tight_layout()
    path = save_path if save_path else f"{Config.IMAGE_PLOT_PREFIX}_{epoch}.png"
    plt.savefig(path)
    plt.close()

# ==========================================
# 5. Sanity Check (起動時チェック)
# ==========================================
def run_sanity_check(decoder, loader):
    print("--- Running Sanity Check ---")
    batch = next(iter(loader))
    batch = batch.to(Config.DEVICE)
    
    # 最初のサンプルのGTだけをデコードしてみる
    gt_slots = batch[0] # [T, S, D]
    
    print(f"Slot Data Stats: Mean={gt_slots.mean():.4f}, Max={gt_slots.max():.4f}")
    if gt_slots.abs().max() < 1e-6:
        print("!!! CRITICAL: Slot data seems to be all zeros. Training will fail. !!!")
        return False
        
    try:
        imgs = decode_slots_to_image(decoder, gt_slots)
        
        # 保存して確認
        plt.figure(figsize=(10, 2))
        for i in range(min(5, len(imgs))):
            plt.subplot(1, 5, i+1)
            plt.imshow(np.clip(imgs[i], 0, 1))
            plt.axis('off')
        plt.suptitle("Sanity Check: GT Reconstruction")
        plt.savefig(Config.SANITY_CHECK_PATH)
        print(f"Sanity check image saved to {Config.SANITY_CHECK_PATH}")
        print("Please check this image immediately. If it is gray/noise, stop training.")
        return True
    except Exception as e:
        print(f"Sanity Check Failed: {e}")
        return False

# ==========================================
# 6. Main
# ==========================================
def main():
    print("=== Start Latent Slot Dynamics Training (Verified) ===")
    
    dataset = SlotPtDataset(Config.SLOT_DATA_DIR, seq_len=Config.SEQ_LEN)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Decoderロード
    decoder = load_frozen_decoder()
    
    # ★起動時チェック★
    if not run_sanity_check(decoder, loader):
        print("Aborting training due to sanity check failure.")
        return

    # Dynamicsモデル
    model = SlotDynamicsTransformer().to(Config.DEVICE)
    print(f"Dynamics Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss() 
    
    history = {'loss': [], 'psnr': []}
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        for batch_slots in pbar:
            batch_slots = batch_slots.to(Config.DEVICE)
            input_slots = batch_slots[:, :-1] 
            target_slots = batch_slots[:, 1:] 
            
            optimizer.zero_grad()
            pred_slots = model(input_slots) 
            loss = criterion(pred_slots, target_slots)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_loss = epoch_loss / len(loader)
        avg_psnr = calculate_psnr(avg_loss)
        
        history['loss'].append(avg_loss)
        history['psnr'].append(avg_psnr)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Latent PSNR: {avg_psnr:.2f} dB")
        
        if (epoch + 1) % 5 == 0:
            save_name = os.path.join(Config.SAVE_DIR, f"dynamics_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_name)
            save_image_comparison(decoder, target_slots, pred_slots, epoch+1)
            save_history_plot(history)

    print("Training Finished.")

if __name__ == "__main__":
    main()