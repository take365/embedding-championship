import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import math
import argparse
from config_models import MODELS
# 日本語フォントの指定
rcParams['font.family'] = 'Meiryo' 


def load_umap_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate_model(X, k_range):
    silhouette, dbi, chi, wcss = [], [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        labels = km.labels_
        silhouette.append(silhouette_score(X, labels))
        dbi.append(davies_bouldin_score(X, labels))
        chi.append(calinski_harabasz_score(X, labels))
        wcss.append(km.inertia_)
    return silhouette, dbi, chi, wcss

def plot_comparison(metric_values, title, ylabel, filename, k_range,output_dir):
    plt.figure(figsize=(10, 6))
    for key, values in metric_values.items():
        plt.plot(k_range, values, label=key)

    base_k = math.sqrt(len(k_range) * 2)
    lower_threshold = round(base_k * 2)
    plt.axvline(x=lower_threshold, color='gray', linestyle='--', linewidth=1)
    plt.text(lower_threshold + 0.5, plt.ylim()[1]*0.9, '下位層開始', rotation=90, color='gray')

    plt.xlabel("クラスタ数 (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / filename)
    plt.close()
def suggest_cluster_range(n_samples: int, level: str) -> range:
    base_k = math.sqrt(n_samples / 2)

    if level == "all":
        # 全体評価用：たとえば k=2〜4倍まで広くカバー
        return range(round(base_k / 4), round(base_k * 4) + 1)
    elif level == "upper":
        return range(round(base_k / 4), round(base_k * 2) - 1)
    elif level == "lower":
        return range(round(base_k * 2), round(base_k * 4) + 1)
    else:
        raise ValueError("level must be 'all', 'upper' or 'lower'")
    
def generate_html_report(metrics, k_range,data_dir,output_dir):
    html = ["<html><head><meta charset='utf-8'><title>モデル比較レポート</title></head><body>"]

    title_path = data_dir / "title.txt"
    title = title_path.read_text(encoding="utf-8").strip() if title_path.exists() else "（タイトル未設定）"
    data_count = 0
    for key in metrics:
        df_path = data_dir / f"{key}.pkl"
        if df_path.exists():
            df = pd.read_pickle(df_path)
            data_count = len(df)
            break
    html.append(f"<h1>{title}（{data_count}件）</h1>")

    base_k = math.sqrt(data_count / 2)
    lower_start = round(base_k * 2)

    model_keys = list(metrics.keys())

    def explanation(metric):
        if metric == "silhouette":
            return ("<p>クラスタ内と他クラスタ間の距離の比較に基づく指標で、-1から+1の範囲のスコアを持ちます。"
                    "<b>スコアが高いほど、データが適切にクラスタに割り当てられている</b>ことを示します。"
                    "<a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>[Wikipedia]</a></p>")
        elif metric == "dbi":
            return ("<p>各クラスタのばらつきとクラスタ間の距離比をもとに計算される指標で、"
                    "<b>スコアは小さいほどクラスタが密集かつ明確に分離されている</b>とされます。"
                    "<a href='https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index' target='_blank'>[Wikipedia]</a></p>")
        elif metric == "chi":
            return ("<p>クラスタ内分散とクラスタ間分散の比率からなる指標です。"
                    "<b>値が大きいほどクラスタの分離が良好</b>とされます。"
                    "<a href='https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index' target='_blank'>[Wikipedia]</a></p>")
        elif metric == "wcss":
            return ("<p>クラスタ内のばらつき（各点と重心の距離の二乗和）を示す指標で、"
                    "<b>スコアが小さいほどクラスタ内の凝集性が高い</b>とされます。"
                    "<br>この指標を用いたエルボー法により、最適なクラスタ数を視覚的に決定できます。"
                    "<a href='https://en.wikipedia.org/wiki/Elbow_method_(clustering)' target='_blank'>[Elbow Method]</a></p>")
        return ""

    for metric in ["silhouette", "dbi", "chi", "wcss"]:
        html.append(f"<h2>{metric.upper()} 比較</h2>")
        html.append(explanation(metric))
        if metric == "silhouette":
            # モデル順の並び（保持した順で安定化）
            model_keys = list(MODELS.keys())

            html.append("<h3>総合評価</h3>")
            html.append("<table border='1'>")

            # 見出し行（モデル名）
            header_row_1 = "<tr><th></th>"
            for key in model_keys:
                model_name = MODELS[key]["model_id"]
                header_row_1 += f"<th colspan='2'>{model_name}</th>"
            header_row_1 += "</tr>"
            html.append(header_row_1)

            # 見出し行2（クラスタ数、スコア、注釈）
            header_row_2 = "<tr><th>種別</th>"
            for _ in model_keys:
                header_row_2 += "<th>クラスタ数</th><th>スコア</th>"
            header_row_2 += "</tr>"
            html.append(header_row_2)

            # 行データを3段まとめて出す（全体・上位層・下位層）
            row_types = ["全体", "上位層", "下位層"]
            for row_type in row_types:
                row_html = f"<tr><td>{row_type}</td>"
                for key in model_keys:
                    scores = metrics[key]["silhouette"]
                    k_values = list(k_range)

                    # モデルごとのデータ件数を取得
                    df_path = data_dir / f"{key}.pkl"
                    df = pd.read_pickle(df_path)
                    local_data_count = len(df)

                    # 各モデルごとの正しい範囲を取得（ここが重要な修正点）
                    upper_k_range = suggest_cluster_range(local_data_count, "upper")
                    lower_k_range = suggest_cluster_range(local_data_count, "lower")
                    # 全体のベスト（全範囲）
                    all_k_best = max(zip(k_values, scores), key=lambda x: x[1])

                    # 上位層だけに絞ったスコア群
                    upper_scores = [(k, s) for k, s in zip(k_values, scores) if k in upper_k_range]
                    lower_scores = [(k, s) for k, s in zip(k_values, scores) if k in lower_k_range]

                    if row_type == "全体":
                        k_best, score = all_k_best
                    elif row_type == "上位層":
                        k_best, score = max(upper_scores, key=lambda x: x[1]) if upper_scores else ("-", 0)
                    elif row_type == "下位層":
                        k_best, score = max(lower_scores, key=lambda x: x[1]) if lower_scores else ("-", 0)

                    row_html += f"<td>{k_best}</td><td>{f'{score:.3f}' if isinstance(score, (int, float)) else score}</td>"
                row_html += "</tr>"
                html.append(row_html)

            html.append("</table><br>")

        html.append("<table><tr>")
        html.append(f"<td valign='top'><img src='{metric}_compare.png' width='800'></td>")
        html.append("<td><table border='1'>")

        max_scores = {key: max(metrics[key][metric]) for key in MODELS}
        min_scores = {key: min(metrics[key][metric]) for key in MODELS} if metric in ["dbi", "wcss"] else {}

        if metric == "silhouette":
            html.append("<tr><th>クラスタ数</th>")
            for key in MODELS:
                html.append(f"<th>{key} スコア</th><th>{key} 補足</th>")
            html.append("</tr>")

            annot_maps = {}
            for key in MODELS:
                scores = metrics[key][metric]
                annot_map = {}
                best_value = -float("inf")
                best_k = None
                for idx_i, k_i in enumerate(k_range):
                    current = scores[idx_i]
                    if current > best_value:
                        best_value = current
                        best_k = k_i
                    last_j = None
                    for idx_j in range(idx_i + 1, len(k_range)):
                        comp = scores[idx_j]
                        k_j = list(k_range)[idx_j]
                        if current > comp:
                            last_j = k_j
                            continue
                        else:
                            break
                    if last_j is not None and k_i == best_k:
                        annot_map[k_i] = last_j
                annot_maps[key] = annot_map

            for i, k in enumerate(k_range):
                if k == lower_start:
                    colspan = 1 + len(MODELS) * (2 if metric == "silhouette" else 1)
                    html.append(f"<tr><td colspan='{colspan}' style='text-align:center; font-weight:bold;'>▼ 下位層 ▼</td></tr>")
    
                row = f"<tr><td>{k}</td>"
                for key in MODELS:
                    score = metrics[key][metric][i]
                    note = ""
                    style = ""
                    if score == max_scores[key]:
                        style = " style='background-color:#90ee90'"
                        note = "最高値"
                    else:
                        if k in annot_maps[key]:
                            style = " style='background-color:#e0ffe0'"
                            note = f"k≦{annot_maps[key][k]}で最高"
                    row += f"<td{style}>{f'{score:.3f}'}</td><td>{note}</td>"
                row += "</tr>"
                html.append(row)
        else:
            html.append("<tr><th>クラスタ数</th>")
            for key in MODELS:
                html.append(f"<th>{key}</th>")
            html.append("</tr>")
            for i, k in enumerate(k_range):
                row = f"<tr><td>{k}</td>"
                for key in MODELS:
                    val = metrics[key][metric][i]
                    if metric in ["dbi", "wcss"]:
                        style = " style='background-color:#90ee90'" if val == min_scores[key] else ""
                    else:
                        style = " style='background-color:#90ee90'" if val == max_scores[key] else ""
                    row += f"<td{style}>{f'{val:.3f}'}</td>"
                row += "</tr>"
                html.append(row)


        html.append("</table></td></tr></table><br>")
    html.append("<p>クラスタ数の範囲は以下のように定義されています：</p>"
                "<ul>"
                "<li><b>上位層：</b> √(n/2) の 1/4倍 ～ 4倍</li>"
                "<li><b>下位層：</b> √(n/2) の 2倍 ～ 4倍</li>"
                "</ul>"
                "<p>ここで n は意見数（データ数）です。</p>")

    html.append("</body></html>")

    with open(output_dir / "compare_models.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    args = parser.parse_args()

    data_dir = Path(f"inputs/{args.dataset}")
    # === 設定 ===
    output_dir = Path(f"outputs/{args.dataset}")
    output_dir.mkdir(exist_ok=True)

    all_metrics = {}
    any_path = next((data_dir / f"{key}_umap.pkl" for key in MODELS if (data_dir / f"{key}_umap.pkl").exists()), None)
    if not any_path:
        raise RuntimeError("UMAP済みファイルが見つかりません。")

    df = pd.read_pickle(any_path)
    n_samples = len(df)
    k_range = suggest_cluster_range(n_samples, "all")

    for key in MODELS:
        path = data_dir / f"{key}_umap.pkl"
        if not path.exists():
            print(f"⚠️ スキップ: {key}_umap.pkl が存在しません")
            continue

        print(f"📊 Evaluating {key}...")
        X = load_umap_embeddings(path)
        sil, dbi, chi, wcss = evaluate_model(X, k_range)
        all_metrics[key] = {
            "silhouette": sil,
            "dbi": dbi,
            "chi": chi,
            "wcss": wcss
        }
    for metric in ["silhouette", "dbi", "chi", "wcss"]:
        metric_data = {key: all_metrics[key][metric] for key in all_metrics}
        ylabel = metric.upper()
        plot_comparison(metric_data, f"{metric.upper()} Comparison", ylabel, f"{metric}_compare.png", k_range, output_dir)

    generate_html_report(all_metrics, k_range, data_dir, output_dir)
    print("✅ モデル比較グラフ出力完了")

if __name__ == "__main__":
    main()
