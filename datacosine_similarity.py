import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

# === Embedding ベクトル読み込みと比較 ===
df1 = pd.read_pickle("data1/mpnet_0_1_gpu.pkl")
df2 = pd.read_pickle("data1/mpnet_0_2_gpu.pkl")

vecs1 = np.vstack(df1["embedding"].values)
vecs2 = np.vstack(df2["embedding"].values)

assert vecs1.shape == vecs2.shape, "Embedding ベクトル数または次元数が一致しません"

cos_sims_emb = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in zip(vecs1, vecs2)]
exact_match_emb = [1 if np.array_equal(v1, v2) else 0 for v1, v2 in zip(vecs1, vecs2)]

# === UMAP変換（random_state=0で再現性保証） ===
umap_model = UMAP(n_components=2, random_state=0)
umap1 = umap_model.fit_transform(vecs1)

umap_model = UMAP(n_components=2, random_state=0)
umap2 = umap_model.fit_transform(vecs2)

# 比較
cos_sims_umap = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in zip(umap1, umap2)]
exact_match_umap = [1 if np.array_equal(v1, v2) else 0 for v1, v2 in zip(umap1, umap2)]

# === CSV出力 ===
output_df = pd.DataFrame({
    "index": range(len(cos_sims_emb)),
    "embedding_exact_match": exact_match_emb,
    "embedding_cosine_similarity": cos_sims_emb,
    "umap_exact_match": exact_match_umap,
    "umap_cosine_similarity": cos_sims_umap,
})

output_df.to_csv("data1/embedding_umap_comparison_mpnet.csv", index=False, encoding="utf-8-sig")

# （必要ならUMAPベクトルも保存）
with open("data1/mpnet_0_1_gpu_umap_recalc.pkl", "wb") as f:
    pickle.dump(umap1, f)
with open("data1/mpnet_0_2_gpu_umap_recalc.pkl", "wb") as f:
    pickle.dump(umap2, f)

print("✅ 出力完了: data1/embedding_umap_comparison_mpnet.csv")
