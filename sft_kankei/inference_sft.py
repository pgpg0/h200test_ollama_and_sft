import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

# --- 0. 出力ディレクトリの作成 ---
output_dir = "comparison_outputs_32b"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to the '{output_dir}' directory.")

# --- 1. データセットの読み込みと準備 ---
print("Loading and preparing dataset...")
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling_singleturn", split="train")
split = dataset.train_test_split(test_size=0.2, seed=42)
eval_dataset = split["test"]

num_samples = 3
random_samples = eval_dataset.shuffle(seed=42).select(range(num_samples))
print(f"Randomly selected {num_samples} samples from eval_dataset.")

# --- 2. モデルのパスと設定 ---
base_model_name = "Qwen/Qwen3-32B"
finetuned_model_path = "/home/ubuntu/client/Data_azami/code/sft/qwen3-32b-function_calling-V0-think"
use_flash_attention_2 = False


# --- 3. 応答生成とモデル読み込みの準備 ---
def generate_response(model, tokenizer, messages):
    """モデルの応答を生成する関数"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text


def load_model_and_tokenizer(model_path, is_base_model=False):
    """モデルとトークナイザーを読み込む関数"""
    print(f"\nLoading model from: {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager",
    )
    tokenizer_path = finetuned_model_path if not is_base_model else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("Loading complete.")
    return model, tokenizer


# --- 4. 比較実行 ---
if __name__ == "__main__":
    # 事前にプロンプトと正解データを抽出
    prepared_data = []
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

    # --- 全モデルの応答を先に生成（メモリ効率化のため） ---
    
    # ベースモデルの応答を生成
    print("\n" + "#"*10 + " Generating responses from BASE MODEL " + "#"*10)
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, is_base_model=True)
    base_model.eval()
    base_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            print(f"Generating for sample {i+1}...")
            base_responses.append(generate_response(base_model, base_tokenizer, data["prompt"]))
    
    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ファインチューニング済みモデルの応答を生成
    print("\n" + "#"*10 + " Generating responses from FINE-TUNED MODEL " + "#"*10)
    ft_model, ft_tokenizer = load_model_and_tokenizer(finetuned_model_path)
    ft_model.eval()
    ft_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            print(f"Generating for sample {i+1}...")
            ft_responses.append(generate_response(ft_model, ft_tokenizer, data["prompt"]))

    del ft_model, ft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()


    # --- 結果の表示とファイル書き込み ---
    print("\n\n" + "#"*10 + " COMPARISON RESULTS " + "#"*10)
    for i, data in enumerate(prepared_data):
        sample_num = i + 1
        print("\n" + "="*20 + f" SAMPLE {sample_num}/{num_samples} " + "="*20)

        # 1. プロンプトの表示と保存
        print("\n--- 1. INPUT PROMPT ---")
        prompt_text = ""
        for msg in data['prompt']:
            line = f"[{msg['role']}]\n{msg['content']}"
            print(line)
            prompt_text += line + "\n\n"
        with open(f"{output_dir}/sample_{sample_num}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # 2. 正解データの表示と保存
        print("\n--- 2. GROUND TRUTH (Correct Answer) ---")
        print(data['truth'])
        with open(f"{output_dir}/sample_{sample_num}_ground_truth.txt", "w", encoding="utf-8") as f:
            f.write(data['truth'])

        # 3. ベースモデル出力の表示と保存
        print("\n--- 3. BASE MODEL OUTPUT ---")
        print(base_responses[i])
        with open(f"{output_dir}/sample_{sample_num}_base_model.txt", "w", encoding="utf-8") as f:
            f.write(base_responses[i])

        # 4. ファインチューニング済みモデル出力の表示と保存
        print("\n--- 4. FINE-TUNED MODEL OUTPUT ---")
        print(ft_responses[i])
        with open(f"{output_dir}/sample_{sample_num}_finetuned_model.txt", "w", encoding="utf-8") as f:
            f.write(ft_responses[i])