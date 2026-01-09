import torch
import torch.nn as nn
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2

# ==========================================
# 1. 設定 (学習時と合わせる)
# ==========================================
class Config:
    # --- ファイル指定 (ここを変更！) ---
    # 予測したいファイルの番号またはIDを指定してください
    # 例: "00000", "00123", "video_02500" など
    TARGET_FILE_ID = "02500" 

    # --- パス設定 ---
    # テストするSlotデータ(.pt)のフォルダ
    SLOT_DATA_DIR = r"C:\Users\ibmkt\group5\datasets\movi_a\slots_savi_pixels"
    
    # 学習済みSAVi (デコーダ用)
    SAVI_CHECKPOINT = r"./checkpoints/savi_pixels/savi_pixel_ep60.pth"
    
    # ★学習済みDynamicsモデル (ここを最新のpthに変更してください)★
    DYNAMICS_MODEL_PATH = r"./checkpoints/savi_latent_dynamics/dynamics_ep10.pth"
    
    # 結果保存先
    RESULT_DIR = "./results_slot_prediction"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # --- モデル仕様 (学習時と完全に一致させる) ---
    SEQ_LEN = 10         # Positional Embeddingの最大長
    NUM_SLOTS = 11       
    SLOT_DIM = 64        
    IMG_SIZE = (64, 64)  
    
    DYN_D_MODEL = 128
    DYN_NHEAD = 4
    DYN_LAYERS = 4
    DYN_FF_DIM = 512
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. モデル定義 (学習コードと同じ)
# ==========================================
class SlotDynamicsTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(Config.SLOT_DIM, Config.DYN_D_MODEL)
        
        # Positional Embeddings
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
        # x: [B, T, S, D]
        B, T, S, D = x.shape
        h = self.input_proj(x)
        
        # PosEmbは入力の長さTに合わせてスライス
        h = h + self.pos_emb_time[:, :T, :, :] 
        h = h + self.pos_emb_slot
        
        h_flat = h.reshape(B, T * S, -1)
        mask = torch.triu(torch.ones(T * S, T * S) * float('-inf'), diagonal=1).to(x.device)
        h_flat = self.transformer(h_flat, mask=mask)
        
        out = self.output_proj(h_flat)
        return out.reshape(B, T, S, -1)

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
# 3. ユーティリティ関数
# ==========================================
def load_models():
    # Dynamics
    print(f"Loading Dynamics from {Config.DYNAMICS_MODEL_PATH}...")
    if not os.path.exists(Config.DYNAMICS_MODEL_PATH):
        raise FileNotFoundError(f"Dynamics Model not found: {Config.DYNAMICS_MODEL_PATH}")

    dynamics = SlotDynamicsTransformer().to(Config.DEVICE)
    dynamics.load_state_dict(torch.load(Config.DYNAMICS_MODEL_PATH, map_location=Config.DEVICE))
    dynamics.eval()
    
    # Decoder
    print(f"Loading Decoder from {Config.SAVI_CHECKPOINT}...")
    if not os.path.exists(Config.SAVI_CHECKPOINT):
        raise FileNotFoundError(f"SAVi Checkpoint not found: {Config.SAVI_CHECKPOINT}")

    decoder = SpatialBroadcastDecoder(Config.SLOT_DIM, Config.SLOT_DIM, Config.IMG_SIZE).to(Config.DEVICE)
    checkpoint = torch.load(Config.SAVI_CHECKPOINT, map_location=Config.DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    decoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            decoder_dict[k.replace('decoder.', '')] = v
    
    try:
        decoder.load_state_dict(decoder_dict, strict=True)
    except:
        print("Warning: Loading decoder with strict=False")
        decoder.load_state_dict(decoder_dict, strict=False)

    decoder.eval()
    return dynamics, decoder

def decode_slots(decoder, slots):
    # slots: [T, S, D] -> images: [T, H, W, 3]
    T, S, D = slots.shape
    with torch.no_grad():
        flat = slots.reshape(T*S, D)
        recons, masks = decoder(flat)
        recons = recons.reshape(T, S, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = masks.reshape(T, S, 1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        masks = torch.softmax(masks, dim=1)
        combined = torch.sum(recons * masks, dim=1) # [T, 3, H, W]
        combined = combined.permute(0, 2, 3, 1) # HWC
    return combined.cpu().numpy()

def save_prediction_strip(gt_slots, pred_slots, file_name):
    """
    gt_slots: 正解のスロット系列 [T, S, D]
    pred_slots: 予測されたスロット系列 [T, S, D]
    """
    max_len = min(gt_slots.shape[0], pred_slots.shape[0])
    
    gt_imgs = decode_slots(decoder, gt_slots[:max_len])
    pred_imgs = decode_slots(decoder, pred_slots[:max_len])
    
    fig, axes = plt.subplots(2, max_len, figsize=(max_len*1.5, 3.5))
    if max_len == 1: axes = np.array([axes]).T

    for t in range(max_len):
        # GT Row
        axes[0, t].imshow(np.clip(gt_imgs[t], 0, 1))
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_title("Input (GT)")
        else: axes[0, t].set_title(f"GT +{t}")
        
        # Pred Row
        axes[1, t].imshow(np.clip(pred_imgs[t], 0, 1))
        axes[1, t].axis('off')
        if t == 0: 
            axes[1, t].set_title("Input")
        else:
            axes[1, t].set_title(f"Pred +{t}")
            for spine in axes[1, t].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)

    plt.tight_layout()
    save_path = os.path.join(Config.RESULT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# 4. メイン: 自己回帰予測ループ
# ==========================================
if __name__ == "__main__":
    # 1. モデルロード
    dynamics, decoder = load_models()
    
    # 2. テストデータの検索
    all_files = sorted(glob.glob(os.path.join(Config.SLOT_DATA_DIR, "*.pt")))
    if not all_files:
        print(f"No .pt files found in {Config.SLOT_DATA_DIR}")
        exit()

    # 指定されたIDを含むファイルを検索
    target_file = None
    for f in all_files:
        if Config.TARGET_FILE_ID in os.path.basename(f):
            target_file = f
            break
    
    if target_file is None:
        print(f"Error: File containing ID '{Config.TARGET_FILE_ID}' not found.")
        print("Available files (first 5):")
        for f in all_files[:5]:
            print(f" - {os.path.basename(f)}")
        exit()
        
    print(f"Target File Found: {os.path.basename(target_file)}")

    # 3. 推論実行
    file_name = os.path.basename(target_file).replace('.pt', '.png')
    
    # [Total_Frames, S, D]
    gt_tensor = torch.load(target_file, map_location=Config.DEVICE).float()
    
    # --- 自己回帰予測 (Autoregression) ---
    # 最初の1フレームだけを入力としてスタート
    # curr_seq: [1, 1, S, D] (Batch=1, Time=1)
    curr_seq = gt_tensor[0].unsqueeze(0).unsqueeze(0)
    
    # 最大 SEQ_LEN (10) まで予測
    predict_steps = Config.SEQ_LEN - 1 
    
    for _ in range(predict_steps):
        with torch.no_grad():
            out = dynamics(curr_seq)
            next_frame = out[:, -1:, :, :] # [1, 1, S, D]
            curr_seq = torch.cat([curr_seq, next_frame], dim=1)
    
    # [1, T, S, D] -> [T, S, D]
    pred_tensor = curr_seq.squeeze(0)
    
    # 可視化して保存
    save_prediction_strip(gt_tensor, pred_tensor, f"pred_{file_name}")

    print("Done.")