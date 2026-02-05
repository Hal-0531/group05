import os
# ★OpenMPエラー回避設定（必ず一番最初に記述）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import json
from pathlib import Path  # Pathオブジェクトを使用
import av
import numpy as np
import torch
from tqdm import tqdm

# 必要なモジュールがパスに通っている前提です
from cosmos_tokenizer.utils import tensor2numpy
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

# ==========================================
# 設定
# ==========================================
# ★Pathオブジェクトとして定義 (文字列を Path() で囲む)
INPUT_DIR = Path(r"C:\Users\ibmkt\group5\Fase2\1x\data\train_v2.0")
OUTPUT_DIR = Path("./reconstructed_train_v2.0")

# モデル設定
MODEL_NAME = "Cosmos-Tokenizer-DV8x8x8"
DECODER_PATH = Path(r"C:\Users\ibmkt\group5\models\Cosmos-Tokenizer-DV8x8x8\decoder.jit")

# 処理するシャードの範囲 (0から99)
START_RANK = 0
END_RANK = 99

# デコード設定
BATCH_SIZE = 1
FPS = 30
MAX_BATCHES_PER_VIDEO = None 

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. 準備
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    
    if not OUTPUT_DIR.exists():
        print(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DECODER_PATH.exists():
        raise FileNotFoundError(f"Decoder model not found at: {DECODER_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # 2. デコーダのロード
    print("Loading Decoder...")
    try:
        # パスを文字列に変換して渡す (str(DECODER_PATH))
        decoder = CausalVideoTokenizer(checkpoint_dec=str(DECODER_PATH))
        if hasattr(decoder, '_dec_model') and decoder._dec_model is None:
             raise RuntimeError(f"Failed to load decoder model from {DECODER_PATH}")
        print("Decoder initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading decoder: {str(e)}") from e

    # 3. 各シャードの処理ループ
    for rank in range(START_RANK, END_RANK + 1):
        try:
            process_shard(rank, decoder, device)
        except Exception as e:
            print(f"Error processing shard {rank}: {e}")
            import traceback
            traceback.print_exc()
            continue

def process_shard(rank, decoder, device):
    # パス結合部分 (INPUT_DIR / "フォルダ名")
    # ※データセットの構造に合わせて調整してください
    metadata_path = INPUT_DIR / "metadata" / f"metadata_{rank}.json"
    video_bin_path = INPUT_DIR / "videos" / f"video_{rank}.bin"
    output_file = OUTPUT_DIR / f"reconstructed_video_{rank}.mp4"

    print(f"\nProcessing Shard {rank}...")

    # ファイル存在確認
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path} -> Skipping.")
        return
    if not video_bin_path.exists():
        print(f"Video binary not found: {video_bin_path} -> Skipping.")
        return
    
    if output_file.exists():
        print(f"Output file exists. Overwriting: {output_file}")

    # メタデータ読み込み
    with open(metadata_path, "r") as f:
        metadata_shard = json.load(f)

    # shard_num_frames キーがない場合のエラーハンドリング
    total_frames = metadata_shard.get("shard_num_frames")
    if total_frames is None:
        # metadataにキーがない場合、binファイルサイズから逆算
        vid_size = video_bin_path.stat().st_size
        # int32 (4bytes) * 3 * 32 * 32 = 12288 bytes/frame
        # ※CosmosTokenizerの仕様によっては異なる場合があります
        bytes_per_chunk = 4 * 3 * 32 * 32
        num_chunks = vid_size // bytes_per_chunk
        print(f"Warning: 'shard_num_frames' not found. Estimated chunks: {num_chunks}")
        shape = (num_chunks, 3, 32, 32)
    else:
        print(f"Total frames: {total_frames}")
        num_chunks = math.ceil(total_frames / 17)
        shape = (num_chunks, 3, 32, 32)
    
    # memmap読み込み
    try:
        encoded_video_dataset = np.memmap(video_bin_path, dtype=np.int32, mode="r", shape=shape)
    except Exception as e:
        print(f"Failed to load memmap: {e}")
        return

    # Video Writer設定
    container = av.open(str(output_file), mode="w")
    stream = container.add_stream("hevc_nvenc", rate=FPS) 
    stream.width = 256
    stream.height = 256
    stream.pix_fmt = "yuv420p" 

    num_batches = math.ceil(len(encoded_video_dataset) / BATCH_SIZE)
    
    # バッチ処理
    for i in tqdm(range(num_batches), desc=f"Decoding Rank {rank}"):
        if MAX_BATCHES_PER_VIDEO is not None and i >= MAX_BATCHES_PER_VIDEO:
            break

        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(encoded_video_dataset))

        batch_np = encoded_video_dataset[start_idx:end_idx].copy()
        batch = torch.from_numpy(batch_np).to(device)

        with torch.no_grad():
            reconstructed_batch = decoder.decode(batch)

        reconstructed_numpy = tensor2numpy(reconstructed_batch)

        for sequence in reconstructed_numpy:
            for frame in sequence:
                frame = frame.astype(np.uint8)
                av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    del encoded_video_dataset
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()