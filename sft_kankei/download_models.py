from huggingface_hub import snapshot_download

# モデル名
model_id = "openai/gpt-oss-20b"

# 保存先ディレクトリ（好きなパスに変更）
local_dir = "/data/gpt-oss"

# モデルをダウンロード
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 実体ファイルを保存
)
