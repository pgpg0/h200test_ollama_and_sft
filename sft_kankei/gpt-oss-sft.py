from datasets import load_dataset
INPUT_FILE="/home/ubuntu/client/Data_azami/code/makedata/thinking_results_qa.jsonl"
dataset = load_dataset("json", data_files=INPUT_FILE, split="train")
print(dataset)
#dataset=dataset[:1000]

from transformers import AutoTokenizer
model_name_or_path="/data/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# messages = dataset[0]["messages"]
# conversation = tokenizer.apply_chat_template(messages, tokenize=False)
# print(conversation)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

#dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
#dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

print(dataset[0]['text'])
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to(model.device)

# output_ids = model.generate(input_ids, max_new_tokens=2048)
# response = tokenizer.batch_decode(output_ids)[0]
# print(response)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

from trl import SFTConfig

training_args = SFTConfig(
    learning_rate=1e-5,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_length=4096,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="/data/gpt-oss-20b-sft_qa_think_v1_temp_full",
    report_to="trackio",
    push_to_hub=False,
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(training_args.output_dir)

