import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import os

# --- 設定 ---
MODEL_NAME = "Qwen/Qwen3-4B"
DATA_PATH = "/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_community_standard.jsonl"
OUTPUT_DIR = "./qlora-qwen3-toolcalling-community-standard-out"

# 1. 量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# 2. トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=torch.cuda.current_device(),
    trust_remote_code=True
)

# 3. LoRA設定
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_proj", "c_attn", "q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 4. データセットの読み込み (キャッシュをクリアして再読み込み)
print("データセットを読み込みます...")
train_dataset = load_dataset("json", data_files=DATA_PATH, split="train", download_mode="force_redownload")
print(f"読み込み完了。データ件数: {len(train_dataset)}")
print("データサンプル:")
print(train_dataset[0])

# 5. SFTTrainerの設定
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    dataset_text_field="messages",
    packing=False,
    max_seq_length=2048,
)

# 6. トレーナーの初期化
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

# 7. 学習の実行
print("学習を開始します...")
trainer.train()
print("学習が完了しました。")

# 8. モデルの保存
trainer.save_model()
print(f"モデルが {OUTPUT_DIR} に保存されました。")