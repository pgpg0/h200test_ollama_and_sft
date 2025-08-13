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
name = "azami_v2"
INPUT_FILE = "/home/ubuntu/client/Data_azami/code/synthetic_data_output/combined_dataset_50k.jsonl"
SAFE_FORMATTED_FILE = f"/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_truly_final_50k_{name}.jsonl"
MODEL_NAME = "Qwen/Qwen3-4B"
OUTPUT_DIR = f"./qlora-qwen3-toolcalling-truly-final-out_{name}"

# ==============================================================================
#  ステップA: データ整形処理 (バグ修正済み)
# ==============================================================================
print("="*80)
print("ステップA: データフォーマット処理を開始します... (バグ修正版)")
print("="*80)

def parse_custom_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    if not response_content: return []
    tool_calls_list = []
    tool_blocks = response_content.strip().split('\n\n')
    for block in tool_blocks:
        lines = block.strip().split('\n')
        if not lines: continue
        tool_name = lines[0].strip()
        if not tool_name: continue
        arguments_dict = {}
        for arg_line in lines[1:]:
            arg_line = arg_line.strip()
            if arg_line.startswith('- '):
                key_value_part = arg_line[2:]
                if ': ' in key_value_part:
                    key, value = key_value_part.split(': ', 1)
                    arguments_dict[key.strip()] = value.strip()
        arguments_string = json.dumps(arguments_dict, ensure_ascii=False)
        tool_calls_list.append({"name": tool_name, "arguments": arguments_string})
    return tool_calls_list

def convert_dataset_fixed(input_path: str, output_path: str):
    print(f"入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            try: data = json.loads(line)
            except json.JSONDecodeError: continue
            messages = data.get("messages", [])
            if len(messages) < 2: continue
            
            user_message_original, assistant_message_original = messages[0], messages[1]
            response_content = assistant_message_original.get("content", "")
            if not isinstance(response_content, str): response_content = ""
            parsed_tool_calls = parse_custom_tool_calls(response_content.strip())
            
            # ✨✨✨ --- ここが最重要のバグ修正点 --- ✨✨✨
            # userメッセージを、必要なキーだけを持つ新しい辞書として再構築する。
            # これにより、元のデータに含まれる不要なキー('tool_calls'など)が完全に除去される。
            cleaned_user_message = {
                "role": "user",
                "content": user_message_original.get("content", "")
            }
            # ✨✨✨ ----------------------------------- ✨✨✨

            cleaned_assistant_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": parsed_tool_calls
            }

            output_data = {"messages": [cleaned_user_message, cleaned_assistant_message]}
            fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    print("データフォーマット完了。")

# バグを修正した整形処理を実行
convert_dataset_fixed(INPUT_FILE, SAFE_FORMATTED_FILE)


# ==============================================================================
#  ステップB: SFT学習の実行 (変更なし)
# ==============================================================================
print("\n" + "="*80)
print("ステップB: SFT学習を開始します...")
print("="*80)

# QLoRA設定等
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map="auto",#量子化の場合はautoだとErrorになるため、下を利用
    #device_map=torch.cuda.current_device(),
    trust_remote_code=True)
# peft_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["c_proj", "c_attn", "q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"])
# model = get_peft_model(model, peft_config)

# データセットの読み込み
train_dataset = load_dataset("json", data_files=SAFE_FORMATTED_FILE, split="train")

print("クレンジング後のデータサンプル:")
print(train_dataset[0]['messages'])

def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

def update_dataset(example):
    example["text"] = formatting_func(example)
    for field in ["index", "category", "instruction", "input", "output"]:
        example.pop(field, None)
    return example

ds = train_dataset.map(update_dataset)
print(ds["messages"])

# SFTTrainerの設定
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
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