import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

# --- 0. 出力ディレクトリの作成 ---
output_dir = "comparison_outputs_ambiguity"
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved to the '{output_dir}' directory.")

# --- 1. テストデータの準備 (曖昧さレベル別に定義) ---
print("Preparing test cases with different ambiguity levels...")

# 固定のSystem Prompt
system_prompt = """You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
<tools>
[{'type': 'function', 'function': {'name': 'getInventoryLevels', 'description': 'Retrieves the current inventory levels for each store.', 'parameters': {'type': 'object', 'properties': {'api_endpoint': {'type': 'string', 'description': 'The API endpoint to fetch inventory levels.'}}, 'required': ['api_endpoint']}}}, {'type': 'function', 'function': {'name': 'getOrderStatuses', 'description': 'Fetches the status of orders including expected delivery dates.', 'parameters': {'type': 'object', 'properties': {'api_endpoint': {'type': 'string', 'description': 'The API endpoint to fetch order statuses.'}}, 'required': ['api_endpoint']}}}, {'type': 'function', 'function': {'name': 'getShipmentTracking', 'description': 'Provides real-time tracking information for shipments.', 'parameters': {'type': 'object', 'properties': {'api_endpoint': {'type': 'string', 'description': 'The API endpoint to fetch shipment tracking details.'}}, 'required': ['api_endpoint']}}}]
</tools>
For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{'arguments': <args-dict>, 'name': <function-name>}
</tool_call>"""

# 曖昧さレベル別のUser Prompt
ambiguity_levels = {
    "level_1_slightly_ambiguous": {
        "user_prompt": """I'm in charge of the supply chain for our stores and I really need a complete picture of our current operations to stay on top of things. Could you please pull the latest data for our inventory levels, order statuses, and all active shipment tracking? 

All the API details you should need are right here:
- Inventory Info: `https://api.retailcompany.com/inventory/levels`
- Order Statuses: `https://api.retailcompany.com/orders/statuses`
- Shipment Tracking: `https://api.retailcompany.com/shipments/tracking`

Getting this data would be a huge help for our planning.""",
        "ground_truth": """<tool_call>
{'arguments': {'api_endpoint': 'https://api.retailcompany.com/inventory/levels'}, 'name': 'getInventoryLevels'}
</tool_call>
<tool_call>
{'arguments': {'api_endpoint': 'https://api.retailcompany.com/orders/statuses'}, 'name': 'getOrderStatuses'}
</tool_call>
<tool_call>
{'arguments': {'api_endpoint': 'https://api.retailcompany.com/shipments/tracking'}, 'name': 'getShipmentTracking'}
</tool_call>"""
    },
    "level_2_indirect_and_missing": {
        "user_prompt": """To keep our supply chain running smoothly, I need to check on a few things. I have the endpoints for inventory (`https://api.retailcompany.com/inventory/levels`) and for tracking shipments (`https://api.retailcompany.com/shipments/tracking`). Could you fetch the latest data from these systems for me? I also need to check on order statuses to see expected delivery dates, but I can't seem to find that specific endpoint right now.""",
        "ground_truth": """<tool_call>
{'arguments': {'api_endpoint': 'https://api.retailcompany.com/inventory/levels'}, 'name': 'getInventoryLevels'}
</tool_call>
<tool_call>
{'arguments': {'api_endpoint': 'https://api.retailcompany.com/shipments/tracking'}, 'name': 'getShipmentTracking'}
</tool_call> 
(Note: The model should ideally ask for the missing 'getOrderStatuses' endpoint, but the primary task is to call the two available functions.)"""
    },
    "level_3_vague_goal": {
        "user_prompt": """Things have been really hectic with our retail supply chain lately. I need a full status update across the board to make sure we're not about to run into any delivery issues or stock problems. Can you get me the latest operational data so I can get a clear picture of what's going on?""",
        "ground_truth": """(Note: The model should not call any functions, but instead ask for the required API endpoints for inventory, orders, and shipments.)
Example expected response: 'I can help with that. To get the latest data, I'll need the API endpoints for inventory levels, order statuses, and shipment tracking. Could you please provide them?'"""
    }
}

# --- 3. モデルのパスと設定 ---
base_model_name = "Qwen/Qwen3-32B"
finetuned_model_path = "./sft-func-calling-qwen3-32b/final_model"
use_flash_attention_2 = False # Flash Attentionを無効化


# --- 4. 応答生成とモデル読み込みの準備 ---
def generate_response(model, tokenizer, messages):
    """モデルの応答を生成する関数"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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


# --- 5. 比較実行 ---
if __name__ == "__main__":
    # 事前にプロンプトと正解データを準備
    prepared_data = []
    for level_name, data in ambiguity_levels.items():
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["user_prompt"]}
        ]
        prepared_data.append({
            "level": level_name,
            "prompt": prompt_messages,
            "truth": data["ground_truth"]
        })

    # --- 全モデルの応答を先に生成 ---
    print("\n" + "#"*10 + " Generating responses from BASE MODEL " + "#"*10)
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, is_base_model=True)
    base_model.eval()
    base_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            print(f"Generating for {data['level']}...")
            base_responses.append(generate_response(base_model, base_tokenizer, data["prompt"]))
    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "#"*10 + " Generating responses from FINE-TUNED MODEL " + "#"*10)
    ft_model, ft_tokenizer = load_model_and_tokenizer(finetuned_model_path)
    ft_model.eval()
    ft_responses = []
    with torch.no_grad():
        for i, data in enumerate(prepared_data):
            print(f"Generating for {data['level']}...")
            ft_responses.append(generate_response(ft_model, ft_tokenizer, data["prompt"]))
    del ft_model, ft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 結果の表示とファイル書き込み ---
    print("\n\n" + "#"*10 + " AMBIGUITY TEST RESULTS " + "#"*10)
    for i, data in enumerate(prepared_data):
        print("\n" + "="*20 + f" SAMPLE FOR: {data['level']} " + "="*20)
        
        # 1. プロンプト
        print("\n--- 1. INPUT PROMPT ---")
        prompt_text = ""
        for msg in data['prompt']:
            line = f"[{msg['role']}]\n{msg['content']}"
            print(line)
            prompt_text += line + "\n\n"
        with open(f"{output_dir}/{data['level']}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # 2. 正解データ
        print("\n--- 2. GROUND TRUTH (Expected Behavior) ---")
        print(data['truth'])
        with open(f"{output_dir}/{data['level']}_ground_truth.txt", "w", encoding="utf-8") as f:
            f.write(data['truth'])

        # 3. ベースモデル出力
        print("\n--- 3. BASE MODEL OUTPUT ---")
        print(base_responses[i])
        with open(f"{output_dir}/{data['level']}_base_model.txt", "w", encoding="utf-8") as f:
            f.write(base_responses[i])

        # 4. ファインチューニング済みモデル出力
        print("\n--- 4. FINE-TUNED MODEL OUTPUT ---")
        print(ft_responses[i])
        with open(f"{output_dir}/{data['level']}_finetuned_model.txt", "w", encoding="utf-8") as f:
            f.write(ft_responses[i])