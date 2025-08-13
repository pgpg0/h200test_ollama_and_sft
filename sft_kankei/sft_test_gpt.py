from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import json

# --- 1. データ読み込み（JSONL -> Dataset形式） ---
def load_function_calling_dataset(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 各メッセージを `prompt` と `response` に変換
    formatted = []
    for entry in data:
        messages = entry["messages"]
        prompt = ""
        response = ""

        for msg in messages:
            if msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                # assistant応答が tool_calls のみの場合
                if msg.get("tool_calls"):
                    response = f"<|im_start|>assistant\n<tool_calls>\n{json.dumps(msg['tool_calls'], ensure_ascii=False, indent=2)}\n</tool_calls><|im_end|>\n"
                elif msg.get("content"):
                    response = f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        if prompt and response:
            formatted.append({"prompt": prompt, "response": response})

    return Dataset.from_list(formatted)

# --- 2. データセット読み込み ---
dataset = load_function_calling_dataset("/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_toolcall_format.jsonl")

# --- 3. モデルとトークナイザーの準備 ---
model_name = "Qwen/Qwen3-4B"  # Function Calling対応モデルを想定
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
)

# --- 4. SFT設定 ---
sft_config = SFTConfig(
    output_dir="./qwen3-fc-sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-6,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    max_seq_length=2048,
    report_to="none",
    completion_only_loss=False
)

# --- 5. SFT Trainer 実行 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
    formatting_func=lambda example: example["prompt"] + example["response"]
)

trainer.train()
