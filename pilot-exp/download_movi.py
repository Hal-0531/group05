import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import tqdm

def download_file(url, local_path):
    """ファイルをダウンロードし、進捗バーを表示する"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        # 既に同サイズで存在する場合はスキップ
        if os.path.exists(local_path) and os.path.getsize(local_path) == total_size:
            print(f"Skipping (already exists): {os.path.basename(local_path)}")
            return

        with open(local_path, 'wb') as f, tqdm.tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

def main():
    # 保存先ディレクトリの設定
    # TFDSが読み込める形式のフォルダ構造にします
    base_dir = os.path.join(os.path.expanduser("~"), "tensorflow_datasets", "movi_a", "128x128", "1.0.0")
    os.makedirs(base_dir, exist_ok=True)
    print(f"ダウンロード先: {base_dir}")

    # Kubric公開バケットのURL
    bucket_url = "https://storage.googleapis.com/kubric-public"
    prefix = "tfds/movi_a/128x128/1.0.0/"
    
    # XMLリストを取得してファイル一覧を作成
    print("ファイルリストを取得しています...")
    list_url = f"{bucket_url}?prefix={prefix}"
    response = requests.get(list_url)
    
    # XML解析（名前空間の処理を含む）
    files_to_download = []
    try:
        root = ET.fromstring(response.content)
        # XMLの名前空間を取得 (例: {http://doc.s3.amazonaws.com/2006-03-01})
        namespace = {'ns': root.tag.split('}')[0].strip('{')}
        
        for contents in root.findall('ns:Contents', namespace):
            key = contents.find('ns:Key', namespace).text
            # フォルダ自体や無関係なファイルを除外
            if key.endswith("/") or key == prefix:
                continue
            files_to_download.append(key)
    except Exception as e:
        print(f"XML解析エラー: {e}")
        print("手動でのリストアップを試みます...")
        # 予備: 主要なメタデータのみ試す場合（通常はここに来ないはずです）
        files_to_download = [
            f"{prefix}dataset_info.json",
            f"{prefix}features.json"
        ]

    print(f"ダウンロード対象ファイル数: {len(files_to_download)}")
    print("※合計サイズは約12GBです。")

    # ダウンロード実行
    for key in files_to_download:
        file_name = os.path.basename(key)
        local_path = os.path.join(base_dir, file_name)
        download_url = f"{bucket_url}/{key}"
        
        try:
            download_file(download_url, local_path)
        except Exception as e:
            print(f"エラー ({file_name}): {e}")

    print("\n=== ダウンロード完了 ===")
    print("次のステップで load_local_movi.py を実行してください。")

if __name__ == "__main__":
    main()