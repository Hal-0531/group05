import os
import cv2
import torch
import numpy as np

# --- 設定項目 ---
# 動画ファイル
INPUT_VIDEO_PATH = r"C:\\Users\\ibmkt\\worldmodel\\Cosmos-Tokenizer\\36377.webm"
# 出力ファイル\
OUTPUT_TOKEN_PATH = r"C:\\Users\\ibmkt\worldmodel\\Cosmos-Tokenizer\\latents.pt"

# JITファイルのパス（指定されたパス）
PATH_ENCODER_JIT = r"C:\\Users\\ibmkt\\worldmodel\\Cosmos-Tokenizer\\cosmos_tokenizer\\encoder.jit"

# 解像度設定 (16の倍数)
TARGET_HEIGHT = 368
TARGET_WIDTH = 368
# フレーム数 (動画がこれより短い場合は自動調整されます)
NUM_FRAMES = 120

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_video(video_path, max_frames, height, width):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    print(f"動画読み込み開始: {video_path}")
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # リサイズ
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("フレームが読み込めませんでした")

    # フレーム数を8の倍数に調整
    original_len = len(frames)
    crop_len = (original_len // 8) * 8
    
    if crop_len == 0:
         raise ValueError(f"動画が短すぎます。")
            
    if crop_len < original_len:
        print(f"フレーム数調整: {original_len} -> {crop_len} frames")
        frames = frames[:crop_len]

    # Numpy -> Tensor
    video_np = np.array(frames)
    # 値域を [-1, 1] に正規化
    video_tensor = torch.tensor(video_np, dtype=torch.float32) / 127.5 - 1.0
    # (T, H, W, C) -> (B, C, T, H, W)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    
    return video_tensor.to(DEVICE)

def load_jit_model(path):
    print(f"JITモデルをロード中: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
    
    # torch.jit.load を使って直接ロードします（ライブラリ不要）
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

if __name__ == "__main__":
    try:
        # 1. 前処理
        input_tensor = preprocess_video(INPUT_VIDEO_PATH, NUM_FRAMES, TARGET_HEIGHT, TARGET_WIDTH)
        print(f"入力データ形状: {input_tensor.shape}")

        # 2. モデルロード
        encoder = load_jit_model(PATH_ENCODER_JIT)

        # 3. エンコード実行
        with torch.no_grad():
            print("エンコード処理中...")
            # JITモデルに直接入力します
            result = encoder(input_tensor)
            
            # 出力がタプル(latents, dist)の場合とTensor単体の場合に対応
            if isinstance(result, tuple):
                latents = result[0]
            else:
                latents = result
            
            print(f"Latents形状: {latents.shape}")

        # 4. 保存
        torch.save(latents, OUTPUT_TOKEN_PATH)
        print(f"保存完了: {OUTPUT_TOKEN_PATH}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        # エラー発生時に即閉じないように入力待ちにする（必要なら）
        # input("Press Enter to exit...")