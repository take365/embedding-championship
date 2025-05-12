import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from umap import UMAP


def load_embeddings(path):
    df = pd.read_pickle(path)
    return np.vstack(df["embedding"].values)


def evaluate_clustering(X, k_range):
    silhouette_scores = []
    dbi_scores = []
    chi_scores = []
    wcss_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)

        try:
            sil = silhouette_score(X, labels)
        except:
            sil = np.nan
        try:
            dbi = davies_bouldin_score(X, labels)
        except:
            dbi = np.nan
        try:
            chi = calinski_harabasz_score(X, labels)
        except:
            chi = np.nan

        wcss = kmeans.inertia_

        silhouette_scores.append(sil)
        dbi_scores.append(dbi)
        chi_scores.append(chi)
        wcss_values.append(wcss)

    return silhouette_scores, dbi_scores, chi_scores, wcss_values


def plot_scores(k_range, scores, title, ylabel, output_path):
    plt.figure()
    plt.plot(k_range, scores, marker="o")
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def generate_html_report(output_dir, k_range, sils, dbis, chis, wcss, elapsed_sec):
    html_path = os.path.join(output_dir, "score_report.html")

    def highlight_silhouette(k_values, scores):
        annotations = {}
        best_so_far = -np.inf

        for i, (k, s) in enumerate(zip(k_values, scores)):
            note = ""
            best_updated = False
            if s > best_so_far:
                note = f"（k≦{k}で最高）"
                best_so_far = s
                best_updated = True
            next_lower = i + 1 < len(scores) and scores[i + 1] < s
            highlight = best_updated and next_lower
            annotations[k] = (note, highlight, s)
        return annotations

    sil_annots = highlight_silhouette(k_range, sils)

    def make_table(title, scores, annot_map=None):
        rows = "\n".join(
            f"<tr><td>{k}</td><td>{round(score, 4)}</td><td style='background-color:#e0ffe0'>{annot_map[k][0]}</td></tr>"
            if annot_map and annot_map.get(k, (None, False))[1] else
            f"<tr><td>{k}</td><td>{round(score, 4)}</td><td>{annot_map[k][0] if annot_map and k in annot_map else ''}</td></tr>"
            for k, score in zip(k_range, scores)
        )
        return f"""
        <h3>{title}</h3>
        <table border=\"1\">
            <tr><th>クラスタ数</th><th>スコア</th><th>補足</th></tr>
            {rows}
        </table>
        """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang=\"ja\">
<head><meta charset=\"UTF-8\"><title>クラスタ評価レポート</title></head>
<body>
<h1>クラスタリング評価指標レポート</h1>
<p><b>処理時間：</b>{elapsed_sec:.2f} 秒</p>

<h2>Silhouette Score</h2>
<p>クラスタ内と別クラスタとの距離を比較。<b>スコアは高いほど良い</b>。</p>
<img src=\"silhouette_score.png\" width=\"500\">
{make_table("Silhouette Score", sils, sil_annots)}

<h2>Davies-Bouldin Index</h2>
<p>クラスタのばらつきと間距離の比。<b>スコアは小さいほど良い</b>。</p>
<img src=\"dbi_score.png\" width=\"500\">
{make_table("Davies-Bouldin Index", dbis)}

<h2>Calinski-Harabasz Index</h2>
<p>クラスタ間と内部の分散比。<b>スコアは大きいほど良い</b>。</p>
<img src=\"chi_score.png\" width=\"500\">
{make_table("Calinski-Harabasz Index", chis)}

<h2>WCSS（Within-Cluster Sum of Squares）</h2>
<p>クラスタ内ばらつき。<b>スコアは小さいほど良い</b>。エルボー法に利用。</p>
<img src=\"wcss.png\" width=\"500\">
{make_table("WCSS (inertia)", wcss)}

</body>
</html>""")

    print(f"📄 HTMLレポート出力完了: {html_path}")


def main():
    embedding_path = "data/text-embedding-3-large.pkl"
    umap_path = "data/text-embedding-3-large_umap.pkl"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    if os.path.exists(umap_path):
        print("📥 UMAP済みデータを読み込み中...")
        X = pd.read_pickle(umap_path)
    else:
        print("🔄 UMAP次元削減を実行中...")
        X_high = load_embeddings(embedding_path)
        X = UMAP(n_components=2, random_state=42).fit_transform(X_high)
        pd.to_pickle(X, umap_path)

    k_range = range(2, 51)
    sils, dbis, chis, wcss = evaluate_clustering(X, k_range)

    plot_scores(k_range, sils, "Silhouette Score", "Silhouette", os.path.join(output_dir, "silhouette_score.png"))
    plot_scores(k_range, dbis, "Davies-Bouldin Index", "DBI (lower is better)", os.path.join(output_dir, "dbi_score.png"))
    plot_scores(k_range, chis, "Calinski-Harabasz Index", "CHI (higher is better)", os.path.join(output_dir, "chi_score.png"))
    plot_scores(k_range, wcss, "WCSS (inertia)", "WCSS (lower is better)", os.path.join(output_dir, "wcss.png"))

    elapsed = time.time() - start
    generate_html_report(output_dir, k_range, sils, dbis, chis, wcss, elapsed)


if __name__ == "__main__":
    main()
