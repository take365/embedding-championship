# embed_args_openai.py
import os
import pandas as pd
from tqdm import tqdm
from llm import request_to_embed
from pathlib import Path
import argparse

AVAILABLE_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large"
]

def embed_texts(texts, model, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = request_to_embed(batch, model=model, is_embedded_at_local=False)
        embeddings.extend(batch_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="text-embedding-3-large",
                        help="OpenAIの埋め込みモデル名")
    parser.add_argument("--output", default=None, help="保存用ファイル名（例: large）")  # ← 追加
    args = parser.parse_args()

    dataset_dir = Path(f"inputs/{args.dataset}")
    input_csv = dataset_dir / "args.csv"

    output_name = args.output or args.model.replace("/", "-")
    output_pkl = dataset_dir / f"{output_name}.pkl"

    if output_pkl.exists():
        print(f"✅ 既に存在します: {output_pkl.name}（スキップ）")
        return

    df = pd.read_csv(input_csv)
    if "argument" not in df.columns:
        raise ValueError("CSVに 'argument' カラムが必要です")

    texts = df["argument"].astype(str).tolist()
    print(f"📝 {len(texts)} 件のテキストを読み込みました")

    embeddings = embed_texts(texts, model=args.model)
    df["embedding"] = embeddings

    df.to_pickle(output_pkl)
    print(f"✅ 保存完了: {output_pkl}")

if __name__ == "__main__":
    main()
