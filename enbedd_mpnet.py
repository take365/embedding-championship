# run_embed_one.py
from llm import request_to_local_embed

# 対象のテキスト（必要なら引数でも）
text = "これはテストです。AIによる文章のクラスタリングを試します。"
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# エンベディング取得
embedding = request_to_local_embed([text], model_name=model_name)

# 結果表示
print("✅ 埋め込み結果（ベクトル長={}）:".format(len(embedding[0])))
