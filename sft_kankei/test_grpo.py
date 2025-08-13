import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from peft import PeftModel
from rich import print  # 見やすい表示のために rich を使用（pip install rich）

# データセットの読み込み
dataset_id = "AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

# モデルとトークナイザーの読み込み
model_id = "Qwen/Qwen3-4B"
lora_adapter_path = "/home/ubuntu/client/Data_azami/code/po/Qwen3-4B-GRPO-with-think-reward/checkpoint-100"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
trained_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
trained_tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate_with_reasoning(prompt):
    prompt_str = " ".join(entry["content"] for entry in prompt)
    inputs = trained_tokenizer(prompt_str, return_tensors="pt").to(trained_model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = trained_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    end_time = time.time()

    generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    inference_duration = end_time - start_time
    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens

# 複数サンプルに対して推論
num_samples = 5
print(f"[bold cyan]Running inference on {num_samples} test samples...[/bold cyan]\n")

for i in range(num_samples):
    prompt_data = test_dataset["prompt"][i]
    raw_user_input = prompt_data[1]["content"]
    
    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(prompt_data)

    print(f"[bold green]Sample {i+1}[/bold green]")
    print(f"[bold]User Question:[/bold] {raw_user_input}")
    print(f"[bold]Generated Output:[/bold]\n{generated_text}")
    print(f"[bold]Inference Time:[/bold] {inference_duration:.2f} sec")
    print(f"[bold]Generated Tokens:[/bold] {num_generated_tokens}")
    print("-" * 80)
