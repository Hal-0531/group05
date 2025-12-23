import torch
import cv2
import numpy as np
import os

# --- 設定項目 ---
# 入力トークン
INPUT_TOKEN_PATH = r"C:\Users\ibmkt\worldmodel\Cosmos-Tokenizer\latents.pt"
# 出力動画
OUTPUT_VIDEO_PATH = r"C:\Users\ibmkt\worldmodel\Cosmos-Tokenizer\restored.mp4"

# JITファイルのパス
PATH_DECODER_JIT = r"C:\Users\ibmkt\worldmodel\Cosmos-Tokenizer\cosmos_tokenizer\decoder.jit"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_video(tensor, output_path, fps=24):
    print(f"動画書き出し中: {output_path}")
    
    # (B, C, T, H, W) -> (C, T, H, W)
    tensor = tensor.detach().cpu().squeeze(0)
    
    # 【修正箇所】BFloat16形式だとNumPyがエラーを吐くため、Float32に変換
    tensor = tensor.float() 
    
    # [-1, 1] -> [0, 255]
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1.0) * 127.5
    
    # (C, T, H, W) -> (T, H, W, C)
    # ここで .numpy() を呼び出しますが、既に float32 になっているのでエラーになりません
    video_np = tensor.permute(1, 2, 3, 0).numpy().astype(np.uint8)
    
    T, H, W, C = video_np.shape
    
    # mp4v コーデック
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for i in range(T):
        frame = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print("保存完了！")

def load_jit_model(path):
    print(f"JITモデルをロード中: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
    
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

if __name__ == "__main__":
    if not os.path.exists(INPUT_TOKEN_PATH):
        print(f"エラー: トークンファイルが見つかりません: {INPUT_TOKEN_PATH}")
        exit()

    try:
        # 1. モデルロード
        decoder = load_jit_model(PATH_DECODER_JIT)

        # 2. トークン読み込み
        print(f"トークン読み込み: {INPUT_TOKEN_PATH}")
        latents = torch.load(INPUT_TOKEN_PATH, map_location=DEVICE)
        
        # 念のため読み込んだ時点でもBFloat16ならFloat32にしておく手もありますが
        # モデル入力にはBFloat16が望ましいため、ここではそのままにします
        print(f"Latents形状: {latents.shape}")

        # 3. デコード実行
        with torch.no_grad():
            print("デコード処理中...")
            reconstructed = decoder(latents)
            print(f"再構成形状: {reconstructed.shape}")

        # 4. 動画保存
        save_video(reconstructed, OUTPUT_VIDEO_PATH)

    except Exception as e:
        import traceback
        traceback.print_exc()