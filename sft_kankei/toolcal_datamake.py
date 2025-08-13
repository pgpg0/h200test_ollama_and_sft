import json
from pathlib import Path
from typing import List, Dict, Any

# --- 設定 ---
INPUT_FILE = "/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data/generated_tool_calling_dataset_1.jsonl"
OUTPUT_FILE = "/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_community_standard.jsonl"

def parse_custom_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    if not response_content: return []
    tool_calls_list = []
    tool_blocks = response_content.strip().split('\n\n')

    for block in tool_blocks:
        lines = block.strip().split('\n')
        if not lines: continue
        tool_name = lines[0].strip()
        if not tool_name: continue

        arguments_dict = {}
        for arg_line in lines[1:]:
            arg_line = arg_line.strip()
            if arg_line.startswith('- '):
                key_value_part = arg_line[2:]
                if ': ' in key_value_part:
                    key, value = key_value_part.split(': ', 1)
                    arguments_dict[key.strip()] = value.strip()
        
        # ✨✨✨ --- ここが調査結果を反映した最重要の変更点 --- ✨✨✨
        # 引数の辞書を、JSON形式の「文字列」に変換する
        arguments_string = json.dumps(arguments_dict, ensure_ascii=False)
        # ✨✨✨ ----------------------------------------- ✨✨✨

        tool_calls_list.append({
            "name": tool_name,
            "arguments": arguments_string # 文字列として格納
        })
    return tool_calls_list

def convert_dataset(input_path: str, output_path: str):
    print(f"変換を開始します: {input_path} -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            try: data = json.loads(line)
            except json.JSONDecodeError: continue
            messages = data.get("messages", [])
            if len(messages) < 2: continue
            user_message, assistant_message = messages[0], messages[1]
            response_content = assistant_message.get("content", "")
            if not isinstance(response_content, str): response_content = ""
            
            parsed_tool_calls = parse_custom_tool_calls(response_content.strip())
            
            new_messages = [
                {"role": "user", "content": user_message.get("content", "")},
                {"role": "assistant", "content": None, "tool_calls": parsed_tool_calls}
            ]
            output_data = {"messages": new_messages}
            fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    print("変換が完了しました。")

if __name__ == '__main__':
    convert_dataset(INPUT_FILE, OUTPUT_FILE)