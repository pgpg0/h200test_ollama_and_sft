from enum import Enum
from functools import partial
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# --- 固定設定 ---
seed = 42
set_seed(seed)

model_name = "Qwen/Qwen3-32B"  # or "Qwen/Qwen3-4B-Base"
dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"

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
    model_name,
    trust_remote_code=True,
)
tokenizer.add_special_tokens({
    "additional_special_tokens": ChatmlSpecialTokens.list()
})

# --- 前処理（System role除去 + role統一） ---
def preprocess(sample):
    messages = sample["messages"]

    # system role の結合処理
    # if messages[0]["role"] == "system":
    #     system_msg = messages.pop(0)["content"]
    #     messages[0]["content"] = system_msg + "\n" + messages[0]["content"]

    # # ロール名をQwen形式に統一（human→user, model→assistant）
    # for msg in messages:
    #     if msg["role"] == "human":
    #         msg["role"] = "user"
    #     elif msg["role"] == "model":
    #         msg["role"] = "assistant"

    # Chat template 付きでtext化（tokenize=FalseでSFTTrainer用）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True
    )
    return {"text": text}

# --- データセットの準備 ---
dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.map(preprocess, remove_columns=["messages"])
dataset = dataset["train"].train_test_split(0.1)

# （デバッグ用にデータ件数を制限）
dataset["train"] = dataset["train"]
dataset["test"] = dataset["test"]

print(dataset["train"][0])
# --- モデルの準備 ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- 学習設定 ---
training_arguments = SFTConfig(
    output_dir="qwen3-32b-function_calling-V0-think",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=5,
    learning_rate=5e-5,
    max_grad_norm=1.0,
    num_train_epochs=1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    packing=True,
    eval_strategy="epoch",
    save_strategy="no",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    push_to_hub=False,
    max_seq_length=2048,
)

# --- 学習実行 ---
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

print(trainer.evaluate())
print("訓練開始\n\n\n")
trainer.train()
print("訓練終了\n\n\n")
print(trainer.evaluate())
trainer.save_model()
