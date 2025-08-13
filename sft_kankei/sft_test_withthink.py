# ==============================================================================
#                 最終・バグ修正済み 統合SFT学習スクリプト
# ==============================================================================
import json
import os
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# --- グローバル設定 ---
name = "azami_v3_think"
INPUT_FILE = "/home/ubuntu/client/Data_azami/code/synthetic_data_output/with_think_data/generated_thinking_new_format_1.jsonl"
SAFE_FORMATTED_FILE = f"/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_truly_final_50k_{name}.jsonl"
MODEL_NAME = "Qwen/Qwen3-4B"
OUTPUT_DIR = f"./qlora-qwen3-toolcalling-truly-final-out_{name}"

# ==============================================================================
#  ステップB: SFT学習の実行 (変更なし)
# ==============================================================================
print("\n" + "="*80)
print("ステップB: SFT学習を開始します...")
print("="*80)

# QLoRA設定等
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#print(tokenizer.chat_template)

if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True)
# peft_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["c_proj", "c_attn", "q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"])
# model = get_peft_model(model, peft_config)

# データセットの読み込み
def preprocess_for_template(example):
    """
    データセットの1サンプルを受け取り、アシスタントメッセージの形式を
    チャットテンプレートに合わせて変換する関数。
    """
    # 'messages'キーからメッセージのリストを取得
    messages = example['messages']
    
    # リスト内の各メッセージを処理
    for message in messages:
        # 'role'が'assistant'で、かつ'tool_calls'が存在するメッセージを対象とする
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            
            # contentキーが存在し、中身があることを確認
            if 'content' in message and message['content']:
                
                # 1. 'content'の値を新しい'reasoning_content'キーに移動
                message['reasoning_content'] = message['content']
                
                # 2. 元の'content'キーは空文字列にする
                message['content'] = ''
                
    # 変更を加えたmessagesリストで元のサンプルを更新
    example['messages'] = messages
    return example
train_dataset = load_dataset("json", data_files=INPUT_FILE, split="train")

# --- データセット全体に前処理を適用 ---
# `map`関数を使って、すべてのサンプルに対して`preprocess_for_template`関数を実行
processed_dataset = train_dataset.map(preprocess_for_template)


# --- 結果の確認 ---
# 変更前と変更後の最初のサンプルを比較して確認してみましょう
print("--- 変更前のデータサンプル ---")
print(train_dataset[0]['messages'])

print("\n" + "="*50 + "\n")

print("--- 変更後のデータサンプル ---")
print(processed_dataset[0]['messages'])
# print("クレンジング後のデータサンプル:")
# print(train_dataset[0]['messages'])

def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

def update_dataset(example):
    example["text"] = formatting_func(example)
    for field in ["index", "category", "instruction", "input", "output"]:
        example.pop(field, None)
    return example

ds = processed_dataset.map(update_dataset)
print(len(ds))
print("\n\nformatting後")
print(ds["text"][0])
# SFTTrainerの設定
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-6,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    dataset_text_field="text",
    packing=False,
    max_seq_length=33000,
)

# トレーナーの初期化
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=ds,
)

# 学習の実行
print("\nいよいよ学習を開始します...")
trainer.train()
print("学習が完了しました。")

trainer.save_model()
print(f"モデルが {OUTPUT_DIR} に保存されました。")