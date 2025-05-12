import pandas as pd

# Parquetファイルを読み込む
df = pd.read_parquet("ja.stackoverflow.com-00000-of-00001.parquet")
print(df.shape)
# question.body を argument という列名に変更して2000件だけ抽出
df_out = df[["question.body"]].rename(columns={"question.body": "argument"}).head(2000)

# CSVとして保存
df_out.to_csv("output_sample_2000.csv", index=False, encoding="utf-8-sig")
