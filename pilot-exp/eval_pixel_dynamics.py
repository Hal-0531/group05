import torch
import torch.nn as nn
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
class Config:
    # --- ファイル指定 (ここを変更！) ---
    # 予測したいファイルの番号またはIDを指定してください
    TARGET_FILE_ID = "02500" 

    # --- パス設定 ---
    # 生の動画フォルダ
    DATA_FOLDER = r"./datasets/movi_a/videos"
    
    # ★学習済みPixel Transformerモデル (ここを実際のパスに変更してください)★
    MODEL_PATH = r"./checkpoints/pixel_transformer_dynamics/pixel_trans_ep20.pth"
    
    # 結果保存先
    RESULT_DIR = "./results_pixel_prediction"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- モデル仕様 (学習時と完全に一致させる) ---
    IMG_SIZE = 64
    CHANNELS = 3
    FRAMES_PER_SAMPLE = 10
    
    PATCH_SIZE = 8
    D_MODEL = 128
    NHEAD = 4
    LAYERS = 4
    FF_DIM = 512
    DROPOUT = 0.0 # 推論時はDropoutなしが良い

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. モデル定義 (学習コードと同じ)
# ==========================================
class PixelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = Config.PATCH_SIZE
        self.d_model = Config.D_MODEL
        self.img_size = Config.IMG_SIZE
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3, 
            out_channels=self.d_model, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Positional Embeddings
        self.pos_emb_time = nn.Parameter(torch.zeros(1, Config.FRAMES_PER_SAMPLE, 1, self.d_model))
        self.pos_emb_spatial = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.NHEAD,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.LAYERS)
        
        # Output Head
        self.output_proj = nn.Linear(self.d_model, 3 * self.patch_size * self.patch_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        
        # 1. Patchify & Embedding
        x_flat = x.reshape(b * t, c, h, w)
        patches = self.patch_embed(x_flat)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.reshape(b, t, self.num_patches, self.d_model)
        
        # 2. Add Positional Embeddings
        # 推論時にSEQ_LENを超えないようにスライス
        time_limit = min(t, Config.FRAMES_PER_SAMPLE)
        patches = patches[:, :time_limit, :, :]
        patches = patches + self.pos_emb_time[:, :time_limit, :, :]
        patches = patches + self.pos_emb_spatial
        
        # 3. Flatten for Transformer
        x_seq = patches.reshape(b, time_limit * self.num_patches, self.d_model)
        
        # 4. Causal Masking
        seq_len = x_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # 5. Transformer Forward
        out_seq = self.transformer(x_seq, mask=mask)
        
        # 6. Unpatchify
        out_pixels = self.output_proj(out_seq)
        out_pixels = out_pixels.reshape(b, time_limit, self.num_patches, 3 * self.patch_size * self.patch_size)
        out_pixels = out_pixels.reshape(b, time_limit, int(h/self.patch_size), int(w/self.patch_size), 3, self.patch_size, self.patch_size)
        out_pixels = out_pixels.permute(0, 1, 4, 2, 5, 3, 6)
        out_pixels = out_pixels.reshape(b, time_limit, c, h, w)
        
        return torch.sigmoid(out_pixels)

# ==========================================
# 3. ユーティリティ関数
# ==========================================
def load_video(path):
    # 動画読み込み -> Numpy [T, H, W, 3]
    if path.endswith('.npy'):
        return np.load(path)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= Config.FRAMES_PER_SAMPLE: break
    cap.release()
    return np.array(frames)

def load_model():
    print(f"Loading Model from {Config.MODEL_PATH}...")
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")

    model = PixelTransformer().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    return model

def save_prediction_strip(gt_tensor, pred_tensor, file_name):
    """
    gt_tensor:   [T, C, H, W] (Tensor)
    pred_tensor: [T, C, H, W] (Tensor)
    """
    # Tensor -> Numpy -> [T, H, W, C]
    gt_imgs = gt_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    pred_imgs = pred_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    max_len = min(gt_imgs.shape[0], pred_imgs.shape[0])
    
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
            # 予測部分を赤枠で強調
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
    model = load_model()
    
    # 2. テストデータの検索
    all_files = sorted(glob.glob(os.path.join(Config.DATA_FOLDER, "*.avi")))
    if not all_files:
         # 拡張子フォールバック
        all_files = sorted(glob.glob(os.path.join(Config.DATA_FOLDER, "*.*")))
        
    if not all_files:
        print(f"No video files found in {Config.DATA_FOLDER}")
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
        
    print(f"Target Video Found: {os.path.basename(target_file)}")

    # 3. 推論実行
    file_name = os.path.basename(target_file).split('.')[0] + "_pred.png"
    
    # 動画読み込み -> Normalize -> Tensor [T, C, H, W]
    frames_np = load_video(target_file)
    gt_tensor = torch.from_numpy(frames_np).float() / 255.0
    gt_tensor = gt_tensor.permute(0, 3, 1, 2).to(Config.DEVICE)
    
    # --- 自己回帰予測 (Autoregression) ---
    # 最初の1フレームだけを入力としてスタート
    # curr_seq: [1, 1, C, H, W] (Batch=1, Time=1)
    curr_seq = gt_tensor[0].unsqueeze(0).unsqueeze(0)
    
    # モデルの Positional Embedding 最大長 (10) まで予測
    # 1フレーム目は入力なので、残り9フレームを予測する
    predict_steps = Config.FRAMES_PER_SAMPLE - 1 
    
    print("Generating prediction...")
    for _ in range(predict_steps):
        with torch.no_grad():
            # 現在の系列全体を入力
            out = model(curr_seq)
            
            # Transformerは「次のトークン」を予測するよう学習しているため
            # 最後のタイムステップの出力が「次のフレームの予測」になる
            next_frame = out[:, -1:, :, :, :] # [1, 1, C, H, W]
            
            # 予測結果を系列に追加
            curr_seq = torch.cat([curr_seq, next_frame], dim=1)
    
    # [1, T, C, H, W] -> [T, C, H, W]
    pred_tensor = curr_seq.squeeze(0)
    
    # GTの長さも予測長に合わせてトリミング（表示用）
    gt_tensor = gt_tensor[:Config.FRAMES_PER_SAMPLE]

    # 可視化して保存
    save_prediction_strip(gt_tensor, pred_tensor, file_name)

    print("Done.")