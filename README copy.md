# uv 開発環境 README

このリポジトリは、Docker コンテナ内で [uv](https://github.com/astral-sh/uv) を使って Python の依存関係を管理します。
以下では、**コンテナ内での基本的な使い方**をまとめます。

> 以降のコマンドは、すべて Docker コンテナの中で実行している前提です。

---

## 前提

* `pyproject.toml` がリポジトリ直下にある
* `uv` は Docker イメージ内にインストール済み
* カレントディレクトリはプロジェクトルート（`pyproject.toml` がある場所）

```bash
cd /root/work  # 例：プロジェクトルート
```

---

## 1. `uv sync`：依存関係のインストール・同期

### 役割

* `pyproject.toml` と `uv.lock` を元に、

  * 仮想環境（デフォルトでは `.venv`）を作成
  * 依存パッケージをインストール
* 依存関係を更新したあとにも `uv sync` を再度実行して同期します。

### 使い方

```bash
# プロジェクトルートで実行
uv sync
```

初回実行時：

* `.venv/` ディレクトリが作成され、ここにパッケージがインストールされます。

依存関係を変更したあと（例：`uv add` 実行後）も、基本的には再度 `uv sync` をすると安全です。

---

## 2. 仮想環境のアクティベート / デアクティベート

`uv sync` を実行すると、デフォルトで `.venv` という仮想環境が作成されます。
通常の Python venv と同じようにアクティベートできます。

### アクティベート（activate）

```bash
source .venv/bin/activate
```

プロンプトが例えばこんな感じに変わります：

```text
(.venv) root@container:/root/work#
```

この状態では、

```bash
python
pip
```

などのコマンドは `.venv` 内の Python / パッケージを使います。

### デアクティベート（deactivate）

仮想環境から抜けるとき：

```bash
deactivate
```

これでシステム側の Python（または uv の設定に応じた Python）に戻ります。

> メモ：
> `uv run python main.py` のように `uv run` を使う場合、明示的に `activate` しなくても
> 自動で仮想環境を使って実行されます。

---

## 3. `uv add`：依存パッケージの追加

新しいライブラリを追加するときは、`pip install` ではなく **`uv add`** を使います。
`uv add` は以下を自動で行います：

* `pyproject.toml` の `dependencies` への追記
* `uv.lock` の更新
* 必要であればインストール（+ 後述の `uv sync`）

### 通常の依存関係を追加する場合

```bash
uv add requests
```

これにより：

* `requests` が `pyproject.toml` の `[project] dependencies` に追記
* `uv.lock` 更新
* `.venv` にインストール

### 開発用依存関係（dev-dependencies）を追加する場合

テスト用など、本番コードでは不要なライブラリは `--dev` で追加します。

```bash
uv add --dev pytest
```

`pyproject.toml` の中では、例えば以下のように `dev-dependencies` 側に入ります（設定による）：

```toml
[tool.uv]
dev-dependencies = ["pytest"]
```

---

## 4. `pyproject.toml` の書き換えについて

`pyproject.toml` は、**プロジェクトの依存関係とメタデータの「単一の真実の場所」**です。
基本方針として：

* 依存関係の追加・削除は **`uv add` / `uv remove`** を使う
* 手動で `dependencies` を書き換えた場合は、**必ず `uv sync` を実行**してロックファイルと環境を同期する

### 依存関係の例

`uv add requests` を実行した後、`pyproject.toml` の該当箇所は例えば次のようになります：

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.32.0",
    "numpy>=2.0.0",
]
```

この `dependencies` のリストを **手で変更した場合**：

1. 変更を保存
2. 以下を実行して仮想環境を更新：

   ```bash
   uv sync
   ```

### 手動編集時の注意

* バージョン指定ミスやタイポがあると、`uv sync` 時にエラーになります。
* レビュープロセス（Pull Request）で `pyproject.toml` の diff を確認するようにしてください。
* 通常のフローでは「**uv コマンド経由でのみ依存関係を変える**」ことを推奨します。

---

## 5. 典型的な開発フローまとめ

1. （初回）リポジトリを clone し、Docker コンテナに入る

2. コンテナ内でプロジェクトルートへ移動：

   ```bash
   cd /root/work  # 例
   ```

3. 依存関係をインストール：

   ```bash
   uv sync
   ```

4. 必要なら仮想環境をアクティベート：

   ```bash
   source .venv/bin/activate
   ```

5. 開発・実行：

   ```bash
   python main.py        # など
   # または
   uv run python main.py
   ```

6. 新しいライブラリが必要になったら：

   ```bash
   uv add some-package        # 本番依存
   uv add --dev some-tool     # 開発用依存
   ```

7. 作業終了時に仮想環境から抜ける：

   ```bash
   deactivate
   ```

---

必要に応じて、この README をプロジェクトに合わせて追記・修正して使ってください。
