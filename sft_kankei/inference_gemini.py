

from peft import PeftConfig
from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# PEFTモデルが保存されているディレクトリ or Hub 上のリポジトリ名
peft_model_id = "./qlora-qwen3-toolcalling-truly-final-out_namiuchi_ver1"

model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto")
# model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.eval()

# 推論は同様

messages = [
    {"role"}
    {'role': 'user', 'content': '来週の火曜日に大阪で顧客訪問があるのですが、その日の午前中に大阪市内から最も近い駅までの移動ルートと所要時間を教えてください。また、その駅から顧客のオフィスまでの道順も確認しておきたいです。', 'tool_cals': '', 'tool_calls': None}
]
prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

ids = model.generate(**inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id)

text = tokenizer.decode(ids[0].cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(text)


