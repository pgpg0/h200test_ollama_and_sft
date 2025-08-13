import json
from datasets import load_dataset, Dataset
from tqdm import tqdm

def create_think_content(tool_calls_raw):
    """
    ツール呼び出し情報から<think>ブロックのコンテンツを生成する。
    """
    if not isinstance(tool_calls_raw, list):
        tool_calls_raw = [tool_calls_raw]

    func_names = []  # ✅ 修正
    for call in tool_calls_raw:
        func_name = call.get("function", {}).get("name")
        if func_name:
            func_names.append(func_name)
    
    if not func_names:
        return "I need to think about what to do next."

    thoughts = "I need to call the following tools: "
    return thoughts + ", ".join(func_names) + "."

def process_glaive_conversation(conversation):
    """
    単一の会話を処理し、Qwen3 SFT形式の複数のサンプルを生成する。
    """
    processed_samples = []  # ✅ 修正
    base_messages = [{"role": "system", "content": "You are a helpful assistant."}]  # ✅ 修正（固定system）
    
    conversation_turns = conversation.get('conversations', [])
    if not conversation_turns:
        return []

    current_turn_messages = []  # ✅ 修正
    for i, turn in enumerate(conversation_turns):
        role = turn['from']
        value = turn['value']

        if role == 'human':
            current_turn_messages.append({"role": "user", "content": value})
        
        elif role == 'gpt':
            if (i + 1 < len(conversation_turns)) and (conversation_turns[i + 1]['from'] == 'function_call'):
                try:
                    tool_calls_raw = json.loads(conversation_turns[i + 1]['value'])
                    if not isinstance(tool_calls_raw, list):
                        tool_calls_raw = [tool_calls_raw]

                    qwen_tool_calls = []  # ✅ 修正
                    for call in tool_calls_raw:
                        qwen_tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"]
                            }
                        })
                    
                    think_content = create_think_content(tool_calls_raw)
                    assistant_content = f"<|im_start|>thought\n{think_content}<|im_end|>\n{json.dumps(qwen_tool_calls, ensure_ascii=False)}"
                    current_turn_messages.append({"role": "assistant", "content": assistant_content})
                except (json.JSONDecodeError, KeyError):
                    current_turn_messages.append({"role": "assistant", "content": value})
            else:
                current_turn_messages.append({"role": "assistant", "content": value})

        elif role == 'observation':
            if (i - 1 >= 0) and (conversation_turns[i - 1]['from'] == 'function_call'):
                try:
                    tool_calls_raw = json.loads(conversation_turns[i - 1]['value'])
                    if not isinstance(tool_calls_raw, list):
                        tool_calls_raw = [tool_calls_raw]
                    
                    tool_responses = []  # ✅ 修正
                    for call in tool_calls_raw:
                        func_name = call.get("function", {}).get("name")
                        if func_name:
                            tool_responses.append({
                                "role": "tool",
                                "name": func_name,
                                "content": value
                            })
                    current_turn_messages.extend(tool_responses)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if current_turn_messages and current_turn_messages[-1]['role'] in ['assistant', 'tool']:
            full_history = base_messages + current_turn_messages
            
            context_for_sample = []  # ✅ 修正
            for idx, msg in enumerate(full_history):
                new_msg = msg.copy()
                if new_msg['role'] == 'assistant' and idx < len(full_history) - 1:
                    if '<|im_start|>thought' in new_msg['content']:
                        end_thought_tag = '<|im_end|>\n'
                        end_pos = new_msg['content'].find(end_thought_tag)
                        if end_pos != -1:
                            new_msg['content'] = new_msg['content'][end_pos + len(end_thought_tag):].strip()
                context_for_sample.append(new_msg)

            processed_samples.append({"messages": context_for_sample})

    return processed_samples

# --- メイン処理 ---
print("データセットをロードしています...")
source_dataset = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
print("ロード完了。")

all_processed_samples = []  # ✅ 修正
print("データ変換処理を開始します...")
for conversation in tqdm(source_dataset.select(range(10000))):  # テスト用に10000件に制限
    samples = process_glaive_conversation(conversation)
    if samples:
        all_processed_samples.extend(samples)
print("変換処理完了。")

final_dataset = Dataset.from_list(all_processed_samples)

print(f"\n生成されたサンプル数: {len(final_dataset)}")
if len(final_dataset) > 0:
    print("\n--- 修正後のサンプルの一例 ---")
    print(json.dumps(final_dataset[-1], indent=2, ensure_ascii=False))

output_filename = "qwen3_sft_data_final.jsonl"
print(f"\nデータを'{output_filename}'に保存しています...")
final_dataset.to_json(output_filename, force_ascii=False, lines=True)
print("保存が完了しました。")
