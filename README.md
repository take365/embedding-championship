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
※記載の内容はAIによる調査に基づくものです、ハルシネーションを含む可能性がありますのでご了承ください。
### 1. text-embedding-3-small
概要: OpenAIが提供する高効率なテキスト埋め込みモデルで、1536次元のベクトルを生成します。前世代のtext-embedding-ada-002と比較して、精度と効率が向上しています。

用途: 一般的なベクトル検索、セマンティック検索、クラスタリングなど。

次元数：1,536

パラメータ数：非公開（小型モデル）

起動メモリ：API利用（ローカル不要）

最大トークン数：8,191

日本語対応：〇

多言語対応：〇

ライセンス：OpenAI独自

その他：第3世代埋め込みの小型版。前世代モデル（ada-002）より高精度かつ低コスト（約1/5）での提供。クラウドAPI経由でのみ利用可能。

参考リンク: 
* [公式解説](https://platform.openai.com/docs/models/text-embedding-3-small)

### 2. text-embedding-3-large
概要: OpenAIの高性能な埋め込みモデルで、3072次元のベクトルを生成します。多言語対応で、特に高精度な検索や分類タスクに適しています。

用途: 高精度な多言語検索、分類、クラスタリングなど。

次元数：3,072

パラメータ数：非公開（大型モデル）

起動メモリ：API利用（ローカル不要）

最大トークン数：8,191

日本語対応：〇

多言語対応：〇

ライセンス：OpenAI独自

その他：高次元・高精度版。small版よりコストは上がるが、出力次元をAPIで任意に圧縮可能。クラウドAPI専用

参考リンク: 
* [公式解説](https://platform.openai.com/docs/guides/embeddings)

### 🏠 ローカルモデル（Hugging Face）
### 3. sentence-transformers/paraphrase-multilingual-mpnet-base-v2
概要: 50以上の言語に対応した多言語モデルで、768次元のベクトルを生成します。セマンティック検索やクラスタリングに適しています。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約2.78億

起動メモリ：約0.5 GB (GPU) / 1.0 GB (CPU)

最大トークン数：128（自動切り捨て）

日本語対応：〇

多言語対応：〇（50言語以上）

ライセンス：Apache 2.0

その他：XLM-RoBERTaベース。並列コーパスで学習し、言語間セマンティック検索に強み。Hugging Face上でローカル利用可。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

### 4. pkshatech/RoSEtta-base-ja
概要: 日本語に特化したテキスト埋め込みモデルで、1024トークンまでの長文に対応し、CPUでも効率的に動作します。

用途: 日本語のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約1.6億（推定）

起動メモリ：約0.3 GB / 0.6 GB

最大トークン数：1,024

日本語対応：〇（日本語特化）

多言語対応：×

ライセンス：Apache 2.0

その他：RoFormerアーキテクチャ＋RoPE。大規模モデルを蒸留し日本語検索向けに最適化。入力文に「query:」「passage:」プレフィックス要。

参考リンク: 
* [Hugging Face](https://huggingface.co/pkshatech/RoSEtta-base-ja)


### 5. sbintuitions/sarashina-embedding-v1-1b
概要: 1.2Bパラメータの日本語LLMをベースにしたモデルで、1792次元のベクトルを生成します。多段階の対照学習により高精度な埋め込みを実現しています。

用途: 日本語のセマンティック検索、文章分類、クラスタリングなど。

次元数：1,792

パラメータ数：約12億

起動メモリ：約2.4 GB / 4.8 GB

最大トークン数：8,192

日本語対応：〇

多言語対応：×

ライセンス：商用不可（独自）

その他：1.2B日本語LLMをベースに対照学習。JMTEBで高評価。Llama系でLast-tokenプーリング。

参考リンク: 
* [Hugging Face](https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b)


### 6.sentence-transformers/distiluse-base-multilingual-cased-v2
概要: DistilBERTベースの多言語対応モデルで、512次元のベクトルを生成します。50以上の言語に対応し、セマンティック検索やクラスタリングに適しています。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

次元数：512

パラメータ数：約1.35億

起動メモリ：約0.27 GB / 0.54 GB

最大トークン数：512（推奨128）

日本語対応：〇

多言語対応：〇（50言語以上）

ライセンス：Apache 2.0

その他：USEを蒸留した軽量版。DistilBERTベース6層。速度重視の多言語埋め込み。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)

### 7. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
概要: MiniLMベースの多言語対応モデルで、384次元のベクトルを生成します。50以上の言語に対応し、軽量で高速な処理が可能です。

次元数：384

パラメータ数：約1.18億

起動メモリ：約0.24 GB / 0.47 GB

最大トークン数：512

日本語対応：〇

多言語対応：〇（50言語以上）

ライセンス：Apache 2.0

その他：MiniLMベースの12層モデル。ベクトル次元が小さく、高速かつ軽量。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

### 8. sentence-transformers/paraphrase-xlm-r-multilingual-v1
概要: XLM-RoBERTaベースの多言語対応モデルで、768次元のベクトルを生成します。100以上の言語に対応し、高精度な埋め込みを提供します。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約2.78億

起動メモリ：約0.56 GB / 1.1 GB

最大トークン数：512

日本語対応：〇

多言語対応：〇（100言語以上）

ライセンス：Apache 2.0

その他：初期のSBERT多言語モデル。XLM-RoBERTa基盤で翻訳対訳データに強み。

参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

### 9. sentence-transformers/LaBSE
概要: Googleが開発した多言語対応モデルで、109の言語に対応し、共通のベクトル空間にマッピングします。

用途: 多言語対応のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約4.71億

起動メモリ：約0.94 GB / 1.9 GB

最大トークン数：256（最大512）

日本語対応：〇

多言語対応：〇（109言語）

ライセンス：Apache 2.0

その他：Google製。翻訳ペア検出に高性能だが、非翻訳文同士の類似度は他モデルに劣る場合あり。
参考リンク: 
* [Hugging Face](https://huggingface.co/sentence-transformers/LaBSE)

### 10. sonoisa/sentence-bert-base-ja-mean-tokens
概要: 日本語に特化したSentence-BERTモデルで、768次元のベクトルを生成します。日本語のセマンティック検索や文章類似度評価に適しています。

用途: 日本語のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約1.11億

起動メモリ：約0.22 GB / 0.44 GB

最大トークン数：512

日本語対応：〇（日本語特化）

多言語対応：×

ライセンス：CC BY-SA 4.0

その他：東北大BERTベース。全トークン平均プーリング。日本語STSデータでチューニング済み。

参考リンク: Hugging Face
* [Hugging Face](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens)


### 11. cl-nagoya/ruri-v3-310m
概要: 名古屋大学の研究グループが開発した日本語特化のModernBERTベースのモデルで、310Mパラメータを持ちます。

用途: 日本語のセマンティック検索、文章類似度評価など。

次元数：768

パラメータ数：約3.15億

起動メモリ：約0.63 GB / 1.26 GB

最大トークン数：8,192

日本語対応：〇（日本語特化）

多言語対応：×

ライセンス：Apache 2.0

その他：ModernBERT-Ja基盤でFlashAttention対応。クエリ／文書向け接頭辞で用途判別。JMTEB平均77.24点。

参考リンク: 
* [Hugging Face](https://huggingface.co/cl-nagoya/ruri-v3-310m)


### 12. pfnet/plamo-embedding-1b
概要: Preferred Networksが開発した日本語特化のテキスト埋め込みモデルで、1Bパラメータを持ちます。日本語の情報検索やクラスタリングに優れた性能を発揮します。

用途: 日本語のセマンティック検索、文章分類、クラスタリングなど。

次元数：2,048

パラメータ数：約10.5億

起動メモリ：約2.1 GB / 4.2 GB

最大トークン数：4,096

日本語対応：〇（日本語特化）

多言語対応：×

ライセンス：Apache 2.0

その他：Preferred Networks製。独自LLM「PLaMo」基盤。JMTEBで3-large超えの性能。クエリ/文書分離エンコード。

参考リンク: 
* [Hugging Face](https://huggingface.co/pfnet/plamo-embedding-1b)
