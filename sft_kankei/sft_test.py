from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import torch

# 1. データ読み込み
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling_singleturn", split="train")

# 2. データ分割
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# 3. モデルとトークナイザー (OPT-350m用)
model_name = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 特殊トークンを追加
special_tokens_dict = {
    "additional_special_tokens": [
        "<tool_call>",
        "</tool_call>",
        "<tools>",
        "</tools>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>"
    ]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} new tokens to tokenizer.")

# モデルの埋め込み層をリサイズ
model.resize_token_embeddings(len(tokenizer))
print(f"Model embedding layer resized to {len(tokenizer)} tokens.")

# 【重要】OPT-350mのトークナイザーにカスタムのチャットテンプレートを設定
# formatting_func で使用している形式に合わせる
# 'else' ブロックを削除して修正
tokenizer.chat_template = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "<|system|>\n{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "<|user|>\n{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>\n{{ message['content'] }}"
        "{% endif %}"
    "{% endfor %}"
    "{{ eos_token }}" # EOSトークンを追加
)

# 4. conversations を学習用テキストに変換 (カスタムデリミタを使用)
def formatting_func(example):
    parts = []
    # apply_chat_templateに渡すためのmessagesリストを構築
    # formatting_funcの目的は、SFTTrainerが内部で適用するchat_templateに渡す形式にデータを変換すること
    # そのため、{"role": "...", "content": "..."} のリストを返すように変更
    # ただし、SFTTrainerのapply_chat_templateは通常、元のdatasetにchat_templateを適用しようとする
    # そのため、ここでのformatting_funcは、元のデータセットのconversations形式を
    # apply_chat_templateが処理できる「message list」形式に変換するように修正する
    
    # --- 以前の formatting_func (文字列を直接返す) ---
    # for turn in example["conversations"]:
    #     role = turn["from"]
    #     text = turn["value"]
    #     if role == "system":
    #         parts.append(f"<|system|>\n{text}")
    #     elif role in ["user", "human"]:
    #         parts.append(f"<|user|>\n{text}")
    #     elif role in ["assistant", "gpt"]:
    #         parts.append(f"<|assistant|>\n{text}")
    # return "\n".join(parts) + tokenizer.eos_token
    # --- 上記はSFTTrainerの内部処理と競合する可能性があるので変更 ---

    # SFTTrainerは内部でformatting_funcの後にchat_templateを適用しようとするため、
    # formatting_funcはChatML互換のメッセージリストを返すのが最も安全
    # 詳細は https://huggingface.co/docs/trl/main/en/sft_trainer#how-the-sfttrainer-works を参照
    # SFTTrainer's `_prepare_dataset` method:
    # 1. Calls `formatting_func` to get `formatted_text` or `messages`.
    # 2. If `formatted_text` is returned, it's tokenized.
    # 3. If `messages` (ChatML format) is returned, `tokenizer.apply_chat_template` is used.
    
    # したがって、formatting_funcは ChatML 形式のメッセージリストを返すようにする。
    # SFTTrainerはこれを受け取って、上で定義した tokenizer.chat_template を適用する。
    
    messages_list = []
    for turn in example["conversations"]:
        role = turn["from"]
        text = turn["value"]
        
        if role == "system":
            messages_list.append({"role": "system", "content": text})
        elif role in ["user", "human"]:
            messages_list.append({"role": "user", "content": text})
        elif role in ["assistant", "gpt"]:
            messages_list.append({"role": "assistant", "content": text})
            
    return {"messages": messages_list} # ChatML形式のメッセージリストを辞書で返す


# 5. トレーニング設定
training_args = SFTConfig(
    output_dir="./sft-func-calling-opt350m",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=False,
    bf16=False,
    report_to="none",
    max_seq_length=1024,
)

# 6. SFTTrainer 構築
# 6. SFTTrainer 構築
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # 【重要】 'tokenizer' ではなく 'processing_class' 引数に渡す
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

# 7. 評価と訓練
print("Before training:", trainer.evaluate())
trainer.train()
print("After training:", trainer.evaluate())

# モデルの保存
trainer.save_model("./sft-func-calling-opt350m/final_model")