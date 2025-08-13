# ✅ Step 2: 必要なライブラリをインポート
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ✅ Step 3: データセットを読み込み
# Hugging Faceからデータセットを読み込みます (今度は正しく処理します)
dataset = load_dataset("empower-dev/function_calling_eval_multi_turn_v0", split="train")
print("データセットの読み込みが完了しました。")
print(dataset)


# ### ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ### ★★★★★ ここが最終的な正しい前処理関数です ★★★★★
# ### ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# データセットの正しい列名 ('input', 'output', 'functions') を使って前処理を行います
def preprocess(example):
    # 1. 各列からデータを取り出す
    input_messages = example['input']
    output_message = example['output']
    functions_list = example['functions']

    # 2. 利用可能なツール（関数）の定義を文字列に変換
    #    これをシステムプロンプトに埋め込むことで、モデルが利用可能なツールを認識できるようになります
    tools_string = json.dumps(functions_list, indent=2)

    # 3. 入力メッセージをコピーして、システムプロンプトを書き換える
    #    (元のデータに影響を与えないように .copy() を使用)
    processed_messages = [msg.copy() for msg in input_messages]

    # 4. システムプロンプト（リストの最初の要素）に関数の定義を追加
    #    Qwenのチャットテンプレートでは、ツール情報はシステムプロンプトにあると効果的です
    system_prompt = processed_messages[0]['content']
    processed_messages[0]['content'] = f"{system_prompt}\n\n## Available Tools:\n{tools_string}"

    # 5. 入力と出力を結合し、1つの完全な会話シーケンスを作成
    full_conversation = processed_messages + [output_message]

    # 6. SFTTrainerが要求する 'messages' 形式で返す
    return {"messages": full_conversation}

# データセット全体に新しい前処理を適用
processed_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names # 元の 'input', 'output', 'functions' 列は不要なため削除
)

print("\nデータの前処理が完了しました。")
print("処理後のデータセットの最初の1件:")
#print(processed_dataset[0]['messages'])


# ✅ Step 4: モデルとトークナイザーの準備
model_name = "Qwen/Qwen3-4B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Step 5: LoRA（PEFT）の設定
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# )

# ✅ Step 6: SFT（学習）の設定
training_args = SFTConfig(
    output_dir="./qwen3-4b-sft-final-correct",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_seq_length=4096, # ツール定義を含むため長めに設定
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    packing=True,
)

# ✅ Step 7: トレーナーを初期化
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    args=training_args,
)

# ✅ Step 8: 学習を開始
trainer.train()

print("🎉 学習が完了しました！")