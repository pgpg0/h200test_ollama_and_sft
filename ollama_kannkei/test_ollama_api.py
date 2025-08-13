import requests
import json

prompt="Pythonのことを教えてください。"
messages = [{"role": "user", "content": prompt}]

def stream_chat_with_ollama(prompt,
                            model_name="azami-model-byazami:latest",
                            host="localhost",
                            port=11434):
    """
    Ollamaモデルとストリーミングチャットするジェネレータ関数。
    各チャンク（文字列）を yield する。
    """
    url = f"http://{host}:{port}/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": True}

    resp = requests.post(url, json=payload, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line.decode("utf-8"))
        if "response" in chunk:
            yield chunk["response"]
        if chunk.get("done"):
            break

def stream_until_chars(prompt, max_chars=600, **kwargs):
    """
    stream_chat_with_ollama をラップして、
    累積文字数が max_chars に達したら切り上げるジェネレータ。
    """
    buffer = ""
    for piece in stream_chat_with_ollama(prompt, **kwargs):
        buffer += piece
        if len(buffer) >= max_chars:
            # 10文字目までを切り出して返して終了
            yield buffer[:max_chars]
            return
    # もし最後まで max_chars に達しなかったら全体を返す
    if buffer:
        yield buffer

if __name__ == "__main__":
    print("=== 10文字までストリーミング ===")
    for truncated in stream_until_chars("Pythonのことを教えてください。"):
        # ここでは必ず1回しかループしません
        print(truncated)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
base_model="Qwen/Qwen3-4B"
adapter_path="/home/ubuntu/client/Data_azami/code/sft/qlora-qwen3-toolcalling-truly-final-out_azami_v2_think/"
use_flash_attention_2 = False
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)

#model = AutoModelForCausalLM.from_pretrained(adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)
# model = model.merge_and_unload()

def generate_response(model, tokenizer, messages="Pythonについて教えてください。"):
    """モデルの応答を生成する関数"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text

# def load_model_and_tokenizer(model_path, is_base_model=False):
#     """モデルとトークナイザーを読み込む関数"""
#     print(f"\nLoading model from: {model_path} ...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager",
#     )
#     tokenizer_path = finetuned_model_path if not is_base_model else base_model_name
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
#     print("Loading complete.")
#     return model, tokenizer
#ft_model, ft_tokenizer = load_model_and_tokenizer(adapter_path)
response_text=generate_response(model,tokenizer,messages)
print(f"直接推論 model:{adapter_path}")
print(response_text)