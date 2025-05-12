# embed_args_local_mpnet.py
import os
import pandas as pd
from tqdm import tqdm
from llm import request_to_local_embed
from pathlib import Path
import argparse

def embed_texts(texts, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = request_to_local_embed(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    args = parser.parse_args()

    dataset_dir = Path(f"inputs/{args.dataset}")
    input_csv = dataset_dir / "args.csv"
    output_pkl = dataset_dir / "paraphrase-multilingual-mpnet-base-v2.pkl"

    df = pd.read_csv(input_csv)
    if "argument" not in df.columns:
        raise ValueError("CSVに 'argument' カラムが必要です")

    texts = df["argument"].astype(str).tolist()
    print(f"📝 {len(texts)} 件のテキストを読み込みました")

    embeddings = embed_texts(texts)
    df["embedding"] = embeddings

    df.to_pickle(output_pkl)
    print(f"✅ 保存完了: {output_pkl}")

if __name__ == "__main__":
    main()
