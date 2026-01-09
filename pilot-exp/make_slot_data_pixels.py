import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ★ 学習コードからモデル定義と設定を読み込み
try:
    from train_slot_pixels import SAVi_Pixels, IMG_SIZE, DEVICE, NUM_SLOTS, SLOT_DIM
except ImportError:
    print("Error: 'train_savi_pixels.py' が見つかりません。同じフォルダに置いてください。")
    exit()

# ==========================================
# 1. 設定
# ==========================================
# ★ 学習済みモデルのパス
CHECKPOINT_PATH = "./checkpoints/savi_pixels/savi_pixel_ep60.pth" 

# 入力動画と出力先
DATA_DIR = "./datasets/movi_a/videos"
OUTPUT_DIR = "./datasets/movi_a/slots_savi_pixels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 全フレーム読み込み用 Dataset
# ==========================================
class FullVideoDataset(Dataset):
    def __init__(self, video_dir, img_size=(64, 64)):
        self.files = sorted(glob.glob(os.path.join(video_dir, "*.avi")))
        self.img_size = img_size
        print(f"Found {len(self.files)} videos to process.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # リサイズ
            if frame.shape[:2] != self.img_size:
                frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
            
            # BGR -> RGB & Normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            # 万が一空のファイルがあった場合のダミー
            return torch.zeros(1, 3, *self.img_size), filename

        # (T, H, W, C) -> Tensor (T, C, H, W)
        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        return video_tensor, filename

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print(f"=== Extracting Slots from {DATA_DIR} ===")
    
    # モデルの準備
    model = SAVi_Pixels().to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 重みのロード
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # 辞書形式か、直接state_dictかを確認してロード
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # データローダー (Batch Size=1 で1動画ずつ確実に処理)
    dataset = FullVideoDataset(DATA_DIR, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Start processing...")
    
    with torch.no_grad():
        for video_tensor, filenames in tqdm(loader):
            video_tensor = video_tensor.to(DEVICE) # (1, T, 3, H, W)
            
            # 推論実行
            # SAViの戻り値: (recon, masks, recons, slots_seq)
            # 私たちが欲しいのは 4番目の `slots_seq` だけです
            _, _, _, slots_seq = model(video_tensor)
            
            # slots_seq shape: (1, T, Slots, Dim)
            
            # 保存
            # バッチサイズ1なので、squeezeして (T, Slots, Dim) にする
            slots_data = slots_seq.squeeze(0).cpu()
            
            # 元のファイル名に対応する .pt ファイルとして保存
            # video_00000.avi -> video_00000.pt
            save_name = os.path.splitext(filenames[0])[0] + ".pt"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            torch.save(slots_data, save_path)

    print("\nDone!")
    print(f"Saved {len(dataset)} slot files to: {OUTPUT_DIR}")
    
    # データの形状確認
    if len(dataset) > 0:
        sample_file = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(dataset.files[0]))[0] + ".pt")
        sample_data = torch.load(sample_file)
        print(f"Sample Shape (Time, Slots, Dim): {sample_data.shape}")
        # 予想: (24, 6, 64) などの形になっているはず

if __name__ == "__main__":
    main()