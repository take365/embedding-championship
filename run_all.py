# run_all.py
import argparse
import subprocess
from pathlib import Path
import sys
from config_models import MODELS

def run_command(command, description):
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} に失敗しました")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="データセット名（例: pubene, pubkan）")
    args = parser.parse_args()

    dataset = args.dataset
    input_dir = Path(f"inputs/{dataset}")

    for key, info in MODELS.items():
        model_id = info["model_id"]
        source = info.get("source", "local")  # 明示されていない場合はローカル扱い
        model_file = input_dir / f"{key}.pkl"
        umap_file = input_dir / f"{key}_umap.pkl"

        # === 埋め込み ===
        if not model_file.exists():
            if source == "openai":
                run_command(
                    f"python embed_args_openai.py {dataset} --model {model_id} --output {key}",
                    f"[{key}] OpenAI埋め込み: {model_id}"
                )
            else:
                run_command(
                    f"python embed_args_local_custom.py {dataset} {model_id} --output {key}",
                    f"[{key}] ローカル埋め込み: {model_id}"
                )
        else:
            print(f"✅ [{key}] 埋め込み済み: {model_file.name}（スキップ）")

        # === UMAP変換 ===
        if not umap_file.exists():
            run_command(
                f"python umap_one.py {dataset} {key}",
                f"[{key}] UMAP変換"
            )
        else:
            print(f"✅ [{key}] UMAP済み: {umap_file.name}（スキップ）")

    # === 評価スクリプト実行 ===
    run_command(
        f"python all_scores.py {dataset}",
        "📊 モデル評価＆HTMLレポート生成"
    )

if __name__ == "__main__":
    main()
