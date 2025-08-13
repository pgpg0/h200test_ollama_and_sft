import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig
from enum import Enum
from functools import partial
import pandas as pd
import torch
import json

# 1. モデルとトークナイザの準備
model_id = "Qwen/Qwen3-4B-Base"


# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# トークナイザのロード
# --- 特殊トークン定義 ---
class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

# --- トークナイザーの準備 ---
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
tokenizer.add_special_tokens({
    "additional_special_tokens": ChatmlSpecialTokens.list()
})

# 2. データセットの準備 (Part 2で作成したデータセットをロード)
# この例では、Hugging Face Hub上の仮のデータセットをロードします
# 実際には、Part 2のワークフローで生成したファイルパスを指定します
# データセットは、各行が 'messages' キーを持つ辞書形式であると仮定します
# 'messages' の値は、{'role': '...', 'content': '...'} のリストです
processed_dataset = load_dataset("json", data_files="qwen3_sft_data_final.jsonl", split="train")

processed_dataset = processed_dataset.train_test_split(0.1)

# （デバッグ用にデータ件数を制限）
# processed_dataset["train"] = processed_dataset["train"]
# processed_dataset["test"] = processed_dataset["test"]

# 4. SFTTrainer の設定
training_arguments = SFTConfig(
    output_dir="./qwen3-4b-base-sft-function-calling-gemini",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=100,
    save_total_limit=3,
    logging_steps=10,
    learning_rate=1e-6,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # bfloat16を有効化
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    eval_strategy="epoch",
    lr_scheduler_type="constant",
    max_seq_length=2048,
    dataset_text_field="messages",
    packing=True, # パッキングを有効化
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    processing_class=tokenizer,
    args=training_arguments,
)

# 5. トレーニングの開始
print(trainer.evaluate())
print("訓練開始\n\n\n")
trainer.train()


# # 6. モデルの保存
# trainer.save_model("./qwen3-4b-sft-function-calling-final")
# # tokenizer.save_pretrained("./qwen3-4b-sft-function-calling-final")