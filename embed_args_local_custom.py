# embed_args_local_custom.py
import os
import pandas as pd
from tqdm import tqdm
from llm import request_to_local_embed
from pathlib import Path
import argparse

def embed_texts(texts, model_name, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = request_to_local_embed(batch, model_name=model_name)
        embeddings.extend(batch_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    parser.add_argument("model_name", help="HuggingFace上のSentenceTransformerモデル名")
    parser.add_argument("--output", default=None, help="保存用ファイル名（例: mpnet）")  # ← 追加
    args = parser.parse_args()

    dataset_dir = Path(f"inputs/{args.dataset}")
    input_csv = dataset_dir / "args.csv"

    # 保存名は指定があれば使う。なければ model_name を使ってファイル名整形
    output_name = args.output or args.model_name.replace("/", "-")
    output_pkl = dataset_dir / f"{output_name}.pkl"

    if output_pkl.exists():
        print(f"✅ 既に存在します: {output_pkl.name}（スキップ）")
        return

    df = pd.read_csv(input_csv)
    if "argument" not in df.columns:
        raise ValueError("CSVに 'argument' カラムが必要です")

    texts = df["argument"].astype(str).tolist()
    print(f"📝 {len(texts)} 件のテキストを読み込みました")

    embeddings = embed_texts(texts, model_name=args.model_name)
    df["embedding"] = embeddings

    df.to_pickle(output_pkl)
    print(f"✅ 保存完了: {output_pkl}")

if __name__ == "__main__":
    main()