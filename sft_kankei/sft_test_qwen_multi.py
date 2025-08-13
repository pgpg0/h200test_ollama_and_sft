import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

# --- 1. データセットの読み込み ---
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling", split="train")

# --- 2. モデルとトークナイザーの準備 ---
model_name = "/home/ubuntu/client/Data_azami/code/sft/Qwen4b-base"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 特殊トークンの追加（これは正しい処理なので変更なし）
special_tokens_dict = {
    "additional_special_tokens": ["<tool_call>", "</tool_call>", "<tools>", "</tools>", "<tool_response>", "</tool_response>"]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

# --- 3. ★★★ 修正後のフォーマット関数 ★★★ ---
def formatting_func(example):
    messages_list = []
    for turn in example["conversations"]:
        role = turn["from"]
        content = turn["value"]
        
        if role == "system":
            messages_list.append({"role": "system", "content": content})
        elif role in ["user", "human"]:
            messages_list.append({"role": "user", "content": content})
        elif role in ["assistant", "gpt"]:
            messages_list.append({"role": "assistant", "content": content})
        
        ### ★ 修正点 1 ★ ###
        # ツールの応答は、意味的に正しい `tool` ロールとして扱う
        elif role == "tool":
            messages_list.append({"role": "tool", "content": content})
            
    return messages_list

# --- 4. データの分割 ---
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# --- 5. トレーニング設定 (SFTConfig) ---
training_args = SFTConfig(
    output_dir="./sft-qwen3-base-multi", # 出力先ディレクトリ名を変更
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=50,
    learning_rate=2e-5,
    
    ### ★ 修正点 2 ★ ###
    # エポック数を1に減らし、ベースモデルの能力を維持しつつ過学習を抑制
    num_train_epochs=3,
    
    logging_steps=10,
    fp16=False,
    bf16=True,
    max_seq_length=4096,
    packing=True,
    
    # 評価を有効にするための設定
    eval_strategy="steps",
    eval_steps=25,
)

# --- 6. SFTTrainerの構築 ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
)

# --- 7. 訓練の実行 ---
print("--- Evaluating before training ---")
print(trainer.evaluate())
print("--- Starting training ---")
trainer.train()
print("--- Evaluating after training ---")
print(trainer.evaluate())