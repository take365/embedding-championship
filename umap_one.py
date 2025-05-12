# umap_one.py
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from umap import UMAP
import argparse

def load_embeddings(path):
    df = pd.read_pickle(path)
    if "embedding" not in df.columns:
        raise ValueError(f"'embedding' 列が見つかりません: {path}")
    return np.vstack(df["embedding"].values)

def save_umap(embedding_array, output_path):
    reducer = UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embedding_array)
    with open(output_path, "wb") as f:
        pickle.dump(reduced, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    parser.add_argument("key", help="モデルキー（例: mpnet, large）")
    args = parser.parse_args()

    dataset = args.dataset
    key = args.key
    data_dir = Path(f"inputs/{dataset}")
    input_path = data_dir / f"{key}.pkl"
    output_path = data_dir / f"{key}_umap.pkl"

    if not input_path.exists():
        print(f"⛔ 入力ファイルが見つかりません: {input_path.name}")
        return

    if output_path.exists():
        print(f"✅ UMAP済み: {output_path.name}（スキップ）")
        return

    print(f"🔄 UMAP変換開始: {key} on {dataset}")
    try:
        X = load_embeddings(input_path)
        save_umap(X, output_path)
        print(f"✅ UMAP保存完了: {output_path.name}")
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()
