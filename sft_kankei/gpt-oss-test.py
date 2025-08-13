from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Your model loading code
model_name_or_path="/data/gpt-oss-20b"
#model = LLM(model=model_name_or_path)

# Add the new system prompt here
system_prompt = "あなたはB2B向け製品の営業担当として、顧客からの問い合わせに対して丁寧かつ正確に回答します。"

prompt="""
以下の商品情報と質問に対する回答を作成してください。
ただし、もし回答に必要な情報が不足していて正確な回答が出来ない場合は、
どのような情報があれば回答できるかを提示してください。

【商品情報】
商品名: 即熱はんだこて
価格: ￥2,998 (税込￥3,298)
特徴:
クイックスイッチで急速加熱が可能。
耐熱キャップ付き。
出力（温度）を「強（90W/560℃）」と「標準（15W/390℃）」の2段階で切り替え可能。
用途例: 「強」はターミナルやケーブル、「標準」はプリント基板など。
【質問】
はんだ付けの初心者です。この製品は、温度が2段階に固定されているようですが、細かく温度を調節できるタイプのはんだこてと比較して、どのようなメリット・デメリットがありますか？初心者でも扱いやすいでしょうか？
"""

# Your chat messages
messages = [{"role": "user", "content": prompt}]
# Your chat messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

# 1. Get the tokenizer from the LLM engine
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

#tokenizer = model.get_tokenizer()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.6, "top_p":None, "top_k":None}

output_ids = model.generate(input_ids, **gen_kwargs)
response = tokenizer.batch_decode(output_ids)[0]
print("ベース出力\n")
print("="*60)
print(response)
print("="*60)

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Load the original model first
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

# # Merge fine-tuned weights with the base model
# peft_model_id = "/data/gpt-oss-20b-sft_qa_think_v1/"
# model = PeftModel.from_pretrained(base_model, peft_model_id)
# model = model.merge_and_unload()

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device)

# # gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

# output_ids = model.generate(input_ids, **gen_kwargs)
# response = tokenizer.batch_decode(output_ids)[0]
# print("SFT後出力\n\n")
# print("="*60)
# print(response)
# print("="*60)


peft_model_id = "/data/gpt-oss-20b-sft_qa_think_v1_temp/"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

output_ids = model.generate(input_ids, **gen_kwargs)
response = tokenizer.batch_decode(output_ids)[0]
print("SFT後出力(テンプレート)\n\n")
print("="*60)
print(response)
print("="*60)