import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

# --- 0. 出力ディレクトリの作成 ---
output_dir = "comparison_outputs_opt350m"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to the '{output_dir}' directory.")

# --- 1. データセットの読み込み ---
print("Loading and preparing dataset...")
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling_singleturn", split="train")
split = dataset.train_test_split(test_size=0.2, seed=42)
eval_dataset = split["test"]
num_samples = 3
random_samples = eval_dataset.shuffle(seed=42).select(range(num_samples))
print(f"Randomly selected {num_samples} samples from eval_dataset.")

# --- 2. モデルのパスと設定 ---
base_model_name = "facebook/opt-350m"
finetuned_model_path = "./sft-func-calling-opt350m/final_model" # OPT-350mのファインチューニング済みモデルのパス

# --- 3. 応答生成とモデル読み込みの準備 ---
def generate_response(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text

def load_opt_model_and_tokenizer(model_path, is_base_model=False):
    """OPTモデル用にカスタマイズした読み込み関数"""
    print(f"\nLoading model from: {model_path} ...")
    
    # モデルの読み込み (設定をOPT用に変更)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 【最重要】ベースモデルの場合、FT時と同じ設定を手動で適用
    if is_base_model:
        print("Applying custom template and special tokens to the base OPT model...")
        special_tokens_dict = {
            "additional_special_tokens": [
                "<tool_call>", "</tool_call>", "<tools>", "</tools>",
                "<|system|>", "<|user|>", "<|assistant|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

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
            "{{ eos_token }}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading complete.")
    return model, tokenizer

# --- 4. 比較実行 ---
if __name__ == "__main__":
    # 事前にプロンプトと正解データを抽出
    prepared_data = []
    # (省略... データ準備部分はQwen3版と同じ)
    for sample in random_samples:
        prompt_messages = []
        ground_truth = ""
        for turn in sample["conversations"]:
            role_map = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant"}
            role = role_map.get(turn["from"])
            if role in ["system", "user"]:
                prompt_messages.append({"role": role, "content": turn["value"]})
            elif role == "assistant":
                ground_truth = turn["value"]
                break
        prepared_data.append({"prompt": prompt_messages, "truth": ground_truth})


    # --- ベースモデルの推論 ---
    base_model, base_tokenizer = load_opt_model_and_tokenizer(base_model_name, is_base_model=True)
    base_model.eval()
    base_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            base_responses.append(generate_response(base_model, base_tokenizer, data["prompt"]))
    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- ファインチューニング済みモデルの推論 ---
    ft_model, ft_tokenizer = load_opt_model_and_tokenizer(finetuned_model_path)
    ft_model.eval()
    ft_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            ft_responses.append(generate_response(ft_model, ft_tokenizer, data["prompt"]))
    del ft_model, ft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 結果の表示とファイル書き込み ---
    # (省略... この部分はQwen3版と同じ)
    print("\n\n" + "#"*10 + " COMPARISON RESULTS (OPT-350m) " + "#"*10)
    for i, data in enumerate(prepared_data):
        sample_num = i + 1
        print("\n" + "="*20 + f" SAMPLE {sample_num}/{num_samples} " + "="*20)

        prompt_text = ""
        for msg in data['prompt']:
            line = f"[{msg['role']}]\n{msg['content']}"
            prompt_text += line + "\n\n"
        with open(f"{output_dir}/sample_{sample_num}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text)

        with open(f"{output_dir}/sample_{sample_num}_ground_truth.txt", "w", encoding="utf-8") as f:
            f.write(data['truth'])

        with open(f"{output_dir}/sample_{sample_num}_base_model.txt", "w", encoding="utf-8") as f:
            f.write(base_responses[i])
            
        with open(f"{output_dir}/sample_{sample_num}_finetuned_model.txt", "w", encoding="utf-8") as f:
            f.write(ft_responses[i])