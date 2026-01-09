

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
TFRECORD_DIR = "./tensorflow_dataset/movi_a/128x128/1.0.0" 
OUTPUT_DIR = "./datasets/movi_a/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MOVi-Aの設定
HEIGHT = 256
WIDTH = 256
FRAMES = 24  # データ分析結果より確定

# ==========================================
# 2. TFRecord パース関数
# ==========================================
def _parse_function(proto):
    # 'video' は 24個のバイト列のリストとして定義されています
    keys_to_features = {
        'video': tf.io.FixedLenFeature([FRAMES], tf.string), 
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['video']

def main():
    # GPU無効化
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tfrecord_files = sorted(glob.glob(os.path.join(TFRECORD_DIR, "*.tfrecord*")))
    if not tfrecord_files:
        print(f"Error: No tfrecord files found in {TFRECORD_DIR}")
        return
        
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    print(f"Extracting to {OUTPUT_DIR} ...")

    video_count = 0
    
    # データセット作成
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(_parse_function)

    for video_bytes_list in tqdm(parsed_dataset):
        # TensorFlow Tensor ([24] string) -> Numpy list of bytes
        frames_bytes = video_bytes_list.numpy()
        
        # 保存先のパス
        save_path = os.path.join(OUTPUT_DIR, f"video_{video_count:05d}.avi")
        
        # VideoWriter設定
        # 解像度は64x64決め打ちですが、もし違う場合は最初のフレームで取得しても良いです
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(save_path, fourcc, 12.0, (WIDTH, HEIGHT))
        
        valid_video = True
        
        for i in range(FRAMES):
            # バイナリデータから画像をデコード (PNG/JPG対応)
            # frombuffer -> imdecode は非常に強力で形式を自動判別します
            img_buf = np.frombuffer(frames_bytes[i], dtype=np.uint8)
            frame_bgr = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            
            if frame_bgr is None:
                print(f"Warning: Could not decode frame {i} in video {video_count}")
                valid_video = False
                break
            
            # リサイズ (念のため)
            if frame_bgr.shape[:2] != (HEIGHT, WIDTH):
                frame_bgr = cv2.resize(frame_bgr, (WIDTH, HEIGHT))
                
            out.write(frame_bgr)
            
        out.release()
        
        if valid_video:
            video_count += 1
        else:
            # 失敗したファイルは削除
            if os.path.exists(save_path):
                os.remove(save_path)

    print(f"Done! Extracted {video_count} videos.")

if __name__ == "__main__":
    main()