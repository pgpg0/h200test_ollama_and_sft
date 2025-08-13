import json
import glob
import os

# 入力JSONファイルのパターン
input_pattern = "/home/ubuntu/client/Data_azami/result/*qa_results.json"
# 出力JSONLファイルパス
output_file = "chat_sft_data_qa.jsonl"

output = []

for input_file in glob.glob(input_pattern):
    print(f"処理中: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for item in data_list:
        product = item.get("product_name", "")
        features = (
            item.get("features_text")
            or item.get("features")
            or item.get("source_features")
            or ""
        )
        qa_pairs = item.get("qa_pairs", {})

        if not qa_pairs:
            print(f"  {os.path.basename(input_file)}: QAなし → スキップ")
            continue

        for i in range(1, len(qa_pairs) // 2 + 1):
            q_key = f"Q{i}"
            a_key = f"A{i}"

            if q_key in qa_pairs and a_key in qa_pairs:
                user_content = (
                    f"以下は商品の説明です。\n"
                    f"商品名: {product}\n"
                    f"商品の特徴: {features}\n\n"
                    f"質問: {qa_pairs[q_key]}"
                )

                assistant_content = qa_pairs[a_key]

                output.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたはB2B向け製品の営業担当として、顧客からの問い合わせに対して丁寧かつ正確に回答します。"
                        },
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })

# JSONLに書き出し
with open(output_file, "w", encoding="utf-8") as f:
    for record in output:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"{len(output)}件のQAペアをchat形式に変換しました。")
