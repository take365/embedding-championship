# 📚 Embedding Championship

## 1. プロジェクト概要

**Embedding Championship** は、意見テキストの集合を複数の基底モデルでベクトル化し、UMAP + KMeansによる次元削減とクラスタリングを経て、モデルごとの結果を見やすいレポートとして可視化、比較評価するパイプラインです。

* OpenAI API (サービス利用)やHuggingFace/ローカルモデル(自前GPU対応)に対応
* Silhouette ScoreやDavies-Bouldin Indexなどの指標で比較
* HTMLのグラフとテーブルで解説付きの評価結果を一括出力

---

## 2. システム構成

パイプラインは、以下のフローで実行されます:

1. **CSV読込**: `inputs/[dataset]/args.csv`の`argument`列を読み込み
2. **基底モデルで基本文を執等のベクトル化**
3. **UMAP** で 2次元に削減
4. **KMeans** の複数k値でクラスタリング & 評価
5. **HTML** レポート出力: `outputs/[dataset]/compare_models.html`

#### フォルダ構成

```
Embedding_Championship/
├── inputs/
│   └── pubene/   # args.csv, 基底モデル別pkl, umap結果
├── outputs/
│   └── pubene/   # compare_models.html, グラフ
├── embed_args_openai.py
├── embed_args_local_custom.py
├── umap_one.py
├── all_scores.py
├── run_all.py
```

---
## 3. 出力ファイルの説明

* `[model].pkl` … 執等結果 (DataFrame: argument + embedding)
* `[model]_umap.pkl` … UMAP削減結果 (numpy array)
* `compare_models.html` … スコア比較 + グラフつきレポート
* `*.png` … 各指標のライングラフ

---

## 4. セットアップ手順

### 必須環境

* Python 3.10+
* `requirements.txt`に基づく依存ライブラリ

```bash
pip install -r requirements.txt
```

### .env 設定

OpenAI / Azure を利用する場合、`.env`にキーやエンドポイント情報を設定

```
OPENAI_API_KEY=...
USE_AZURE=false
...
```

---

## 5. 実行方法

### 一括実行

```bash
python run_all.py pubene
```

基底モデルごとに執等、UMAP、評価レポートまで自動実行されます

### 個別実行

```bash
# OpenAI執等
python embed_args_openai.py pubene --model text-embedding-3-small

# ローカル執等
python embed_args_local_custom.py pubene paraphrase-multilingual-mpnet-base-v2 --output mpnet

# UMAP
python umap_one.py pubene mpnet

# 評価 & HTML生成
python all_scores.py pubene
```
---

## 7. 評価指標の概要

| 指標                                   | 意味                | 良い値             |
| ------------------------------------ | ----------------- | --------------- |
| Silhouette Score                     | 同一クラスタ内の出来の良さ、分離度 | 大きいほど良い (+1に近い) |
| DBI (Davies-Bouldin Index)           | クラスタ内の散らばりの平均比    | 小さいほど良い         |
| CHI (Calinski-Harabasz Index)        | クラスタ間と内部の分散比      | 大きいほど良い         |
| WCSS (Within-Cluster Sum of Squares) | 各点が集群の中心に近いかどうか   | 小さいほど良い         |

詳細はHTMLレポート内に各指標の解説あり。


## 8. 評価指標の概要 モデル一覧
### 🧠 OpenAI APIモデル
### 1. text-embedding-3-small
概要: OpenAIが提供する高効率なテキスト埋め込みモデルで、1536次元のベクトルを生成します。前世代のtext-embedding-ada-002と比較して、精度と効率が向上しています。

用途: 一般的なベクトル検索、セマンティック検索、クラスタリングなど。

参考リンク: 
* [公式解説](https://platform.openai.com/docs/models/text-embedding-3-small)

#### text-embedding-3-large

* 高性能・高次元（3072次元）、多言語対応


### 2. text-embedding-3-large
概要: OpenAIの高性能な埋め込みモデルで、3072次元のベクトルを生成します。多言語対応で、特に高精度な検索や分類タスクに適しています。

用途: 高精度な多言語検索、分類、クラスタリングなど。

参考リンク: 
* [公式解説](https://platform.openai.com/docs/guides/embeddings)

### 🏠 ローカルモデル（Hugging Face）
### 3. sentence-transformers/paraphrase-multilingual-mpnet-base-v2
概要: 50以上の言語に対応した多言語モデルで、768次元のベクトルを生成します。セマンティック検索やクラスタリングに適しています。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

### 4. pkshatech/RoSEtta-base-ja
概要: 日本語に特化したテキスト埋め込みモデルで、1024トークンまでの長文に対応し、CPUでも効率的に動作します。

用途: 日本語のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/pkshatech/RoSEtta-base-ja)


### 5. sbintuitions/sarashina-embedding-v1-1b
概要: 1.2Bパラメータの日本語LLMをベースにしたモデルで、1792次元のベクトルを生成します。多段階の対照学習により高精度な埋め込みを実現しています。

用途: 日本語のセマンティック検索、文章分類、クラスタリングなど。

参考リンク: 
* [Hugging Face](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)


### 6.sentence-transformers/distiluse-base-multilingual-cased-v2
概要: DistilBERTベースの多言語対応モデルで、512次元のベクトルを生成します。50以上の言語に対応し、セマンティック検索やクラスタリングに適しています。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)

### 7. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
概要: MiniLMベースの多言語対応モデルで、384次元のベクトルを生成します。50以上の言語に対応し、軽量で高速な処理が可能です。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

### 8. sentence-transformers/paraphrase-xlm-r-multilingual-v1
概要: XLM-RoBERTaベースの多言語対応モデルで、768次元のベクトルを生成します。100以上の言語に対応し、高精度な埋め込みを提供します。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

### 9. sentence-transformers/LaBSE
概要: Googleが開発した多言語対応モデルで、109の言語に対応し、共通のベクトル空間にマッピングします。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/LaBSE)

### 10. sonoisa/sentence-bert-base-ja-mean-tokens
概要: 日本語に特化したSentence-BERTモデルで、768次元のベクトルを生成します。日本語のセマンティック検索や文章類似度評価に適しています。

用途: 日本語のセマンティック検索、文章類似度評価など。

参考リンク: Hugging Face
* [Hugging Face](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens)


### 11. cl-nagoya/ruri-v3-310m
概要: 名古屋大学の研究グループが開発した日本語特化のModernBERTベースのモデルで、310Mパラメータを持ちます。

用途: 日本語のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/cl-nagoya/ruri-v3-310m)


### 12. pfnet/plamo-embedding-1b
概要: Preferred Networksが開発した日本語特化のテキスト埋め込みモデルで、1Bパラメータを持ちます。日本語の情報検索やクラスタリングに優れた性能を発揮します。

用途: 日本語のセマンティック検索、文章分類、クラスタリングなど。

参考リンク: 
* [Hugging Face](https://huggingface.co/pfnet/plamo-embedding-1b)
