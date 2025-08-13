from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import torch
from peft import LoraConfig

# 1. データ読み込み (変更なし)
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling_singleturn", split="train")

# 2. データ分割 (変更なし)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# 3. モデルとトークナイザー (Qwen3-32B用に変更)
model_name = "Qwen/Qwen3-32B" # <-- 変更: モデル名をQwen3に変更

# --- 高速化のための変更点 ---
# bfloat16は、最新のNVIDIA GPU (Ampere世代以降)で高い性能を発揮します
# torch.compileとFlash Attention 2も同様です
use_flash_attention_2 = False # Flash Attention 2 を使うかどうか

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # <-- 高速化: float32からbfloat16に変更
    device_map="auto",
    trust_remote_code=True,      # <-- 変更: Qwenのようなカスタムモデル構造を読み込むために必要
    attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager", # <-- 高速化
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Qwen3のトークナイザーは、デフォルトでpad_tokenが設定されていない場合がある
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 特殊トークンの追加 (hermesデータセットの形式に合わせるため) ---
# Qwen3は独自の特殊トークンを持っていますが、学習データセットに含まれる
# <tool_call>などを認識させるために追加しておくと安全です。
special_tokens_dict = {
    "additional_special_tokens": [
        "<tool_call>",
        "</tool_call>",
        "<tools>",
        "</tools>",
    ]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
if num_added_toks > 0:
    print(f"Added {num_added_toks} new tokens to tokenizer.")
    # モデルの埋め込み層をリサイズ
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model embedding layer resized to {len(tokenizer)} tokens.")


# --- チャットテンプレートの削除 ---
# Qwen3のトークナイザーは、最適なチャットテンプレートを内部に持っています。
# 手動で設定する代わりに、これを自動的に利用させるのがベストプラクティスです。
# そのため、手動の `tokenizer.chat_template = ...` のブロックは完全に削除します。


# 4. conversations を学習用テキストに変換 (一部修正)
def formatting_func(example):
    # SFTTrainerが `apply_chat_template` を使うためには、
    # 'role'と'content'のキーを持つ辞書のリストが必要です。
    messages_list = []
    for turn in example["conversations"]:
        role = turn["from"]
        text = turn["value"]
        
        # データセットの 'from' の値を 'role' の標準的な名前に変換
        if role == "system":
            messages_list.append({"role": "system", "content": text})
        elif role in ["user", "human"]:
            messages_list.append({"role": "user", "content": text})
        elif role in ["assistant", "gpt"]:
            messages_list.append({"role": "assistant", "content": text})

    # 【重要】辞書ではなく、メッセージのリストそのものを返すように修正
    return messages_list

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

# 5. トレーニング設定 (高速化設定を追加)
training_args = SFTConfig(
    output_dir="./sft-func-calling-qwen3-32b-lora",
    per_device_train_batch_size=4,  # <-- 高速化: メモリが許す限り増やす (例: 1 -> 4)
    gradient_accumulation_steps=4,  # 有効バッチサイズは 4 * 4 = 16
    max_steps=100,
    learning_rate=2e-4, # 大規模モデルでは学習率を少し下げることが多い
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=False,
    bf16=True,                       # <-- 高速化: bfloat16を有効化
    report_to="none",
    max_seq_length=2048,             # Qwen3は長いコンテキストを扱えるため、必要に応じて増やす
    torch_compile=True,              # <-- 高速化: PyTorch 2.x のコンパイル機能を有効化
)

# 6. SFTTrainer 構築 (変更なし)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    # processing_classは不要。SFTTrainerはmodelからtokenizerを推論します。
)

# 7. 評価と訓練 (変更なし)
print("Before training:", trainer.evaluate()) 
trainer.train()
print("After training:", trainer.evaluate())

# モデルの保存 (変更なし)
trainer.save_model("./sft-func-calling-qwen3-32b-lora/final_model")