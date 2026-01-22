import torch
import torchvision
from torchvision import transforms
import os
import glob
import numpy as np

def encode_with_jit(
    encoder_path: str,
    input_dir: str,
    output_dir: str,
    device: str = "cuda"
):
    # 1. 設定と準備
    print(f"使用デバイス: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 2. JITモデルのロード (ライブラリ不要)
    # ここが最大の変更点です。torch.jit.loadで直接読み込みます。
    if not os.path.exists(encoder_path):
        print(f"エラー: モデルファイルが見つかりません -> {encoder_path}")
        return

    print(f"JITモデルをロード中... {encoder_path}")
    try:
        model = torch.jit.load(encoder_path, map_location=device)
        model.eval()
        # NVIDIAのモデルは通常 BFloat16 に最適化されています
        # GPUが古い場合は float16 または float32 に変更してください
        model = model.to(torch.bfloat16) 
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    # 3. 動画ファイルの探索
    video_extensions = ['*.avi', '*.mp4', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print("動画ファイルが見つかりません。")
        return

    print(f"動画数: {len(video_files)}")

    # 4. 処理ループ
    resize_transform = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)

    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        save_name = os.path.splitext(filename)[0] + ".pt"
        save_path = os.path.join(output_dir, save_name)

        try:
            # --- 動画読み込みと前処理 (手動実装) ---
            
            # (T, H, W, C) で読み込み
            vframes, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
            
            # フレーム調整 (24フレーム)
            target_frames = 24
            if vframes.shape[0] > target_frames:
                vframes = vframes[:target_frames]
            elif vframes.shape[0] < target_frames:
                # 足りない場合は最後のフレームを複製
                diff = target_frames - vframes.shape[0]
                last = vframes[-1:]
                vframes = torch.cat([vframes, last.repeat(diff, 1, 1, 1)], dim=0)

            # (T, C, H, W) -> (C, T, H, W) に並べ替え
            vframes = vframes.permute(1, 0, 2, 3)

            # リサイズ (256x256)
            vframes = resize_transform(vframes)

            # 正規化 [0, 255] -> [-1, 1]
            vframes = vframes.float() / 127.5 - 1.0

            # バッチ次元追加: (1, C, T, H, W)
            input_tensor = vframes.unsqueeze(0)

            # GPU & 型変換
            input_tensor = input_tensor.to(device).to(torch.bfloat16)

            # --- エンコード実行 ---
            with torch.no_grad():
                # JITモデルの呼び出し
                # Cosmos Tokenizer (Discrete) の JIT は通常 (indices, codes) を返します
                output_tuple = model(input_tensor)
                
                # 最初の要素が indices です
                indices = output_tuple[0]

            # 結果保存 (CPUに戻してint型で保存)
            # indicesは通常 float型のまま返ってくることがあるため、必要ならキャスト
            # しかし保存時はそのままTensorとして保存するのが無難です
            torch.save(indices.cpu(), save_path)
            
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(video_files)}] {filename} -> {indices.shape}")

        except Exception as e:
            print(f"エラー ({filename}): {e}")

    print("完了しました。")

if __name__ == "__main__":
    # ==========================================
    # パス設定 (ここだけ書き換えてください)
    # ==========================================
    
    # 1. encoder.jit のパス (ダウンロードしたファイルそのもの)
    MODEL_PATH = r"C:\Users\ibmkt\group5\models\Cosmos-Tokenizer-DV8x8x8\encoder.jit"
    
    # 2. 動画フォルダ
    INPUT_DIR = r"C:\Users\ibmkt\group5\datasets\movi_a\videos" 
    
    # 3. 出力先
    OUTPUT_DIR = r"C:\Users\ibmkt\group5\datasets\movi_a\tokens_dv128"
    
    # ==========================================
    
    encode_with_jit(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)