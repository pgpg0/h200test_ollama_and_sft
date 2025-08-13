import json
import re
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 設定項目 ---
OUTPUT_DIR = Path('./synthetic_data_output')
MODEL_NAME = "wao/DeepSeek-R1-Distill-Qwen-32B-Japanese"
MAX_ATTEMPTS = 1
NUM_GPUS = 8 # テスト時は1に設定することを推奨
BASE_PORT = 11435
OUTPUT_FILENAME = "simple_prompt_tool_calling.jsonl"

# --- 課題リスト (単純なものに変更) ---
tasks = [
    "明日の15時から「チーム定例」という予定をカレンダーに入れて。",
    "渋谷駅の近くで、評価の高いカフェを教えて。",
    "今日の夜19時に「友人との夕食」というイベントを作成して。",
    "新宿御苑の周辺で、ランチにおすすめの場所を探してほしい。",
    "来週月曜の朝9時に「プロジェクトキックオフ」の予定を登録する。",
    "東京タワーの近くにあるレストランを検索して。"
]

# --- ツール定義 (4つに絞り込み) ---
TOOL_DEFINITIONS = """
### 利用可能なツールリスト

- tool_name: list-events
  description: 指定した期間内のイベントを一覧表示する
- tool_name: create-event
  description: 新しいイベントを作成する
- tool_name: maps_search_places
  description: テキストクエリを使って場所を検索する
- tool_name: maps_directions
  description: 2地点間のルート案内（道のり、距離、時間）を取得する
"""

# --- プロンプトテンプレート (変更なし) ---
PROMPT_TEMPLATE = """
あなたは優秀なアシスタントです。ユーザーの「課題」を解決するために、どの「ツール」をどのような順番で使うべきか計画を立ててください。

### 課題
{task_prompt}

{tool_definitions}

### 指示
上記の課題を解決するためのワークフローを考え、使用するツール名を順番に提示してください。各ツール名は改行して、以下のようにリスト形式で出力してください。

### 回答形式
tool_name_1
tool_name_2
"""

# --- ヘルパー関数 (変更なし) ---
def clean_llm_output(text: str) -> str:
    return text.strip()

def is_tool_list_valid(text: str) -> bool:
    return bool(text and text.strip())

# --- LLMリクエストと検証を行うワーカー関数 (printにflush=Trueを追加) ---
def process_task_for_tool_calling(task_item, host):
    task_prompt = task_item['task_prompt']
    result_data = task_item['base_result_data'].copy()
    result_data.update({"status": "failed", "attempts": 0, "tool_workflow": None, "last_raw_response": None})

    prompt_for_llm = PROMPT_TEMPLATE.format(task_prompt=task_prompt, tool_definitions=TOOL_DEFINITIONS)

    is_valid = False
    attempts = 0
    while not is_valid and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            response = requests.post(
                f"http://{host}/api/chat",
                json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_for_llm}], "stream": False},
                timeout=1000
            )
            response.raise_for_status()
            raw_response = response.json().get("message", {}).get("content", "")

            # flush=Trueを追加
            print(f"\n--- Attempt {attempts} on {host} ---", flush=True)
            print(f"Task: {task_prompt[:50]}...", flush=True)
            print(f"LLM Raw Response: '{raw_response}'", flush=True)

            result_data["last_raw_response"] = raw_response

            cleaned_response = clean_llm_output(raw_response)
            if is_tool_list_valid(cleaned_response):
                is_valid = True
                result_data["status"] = "success"
                result_data["tool_workflow"] = cleaned_response

        except Exception as e:
            result_data["last_raw_response"] = f"Error: {e}"
            # flush=Trueを追加
            print(f"\n--- ERROR on {host} (Attempt {attempts}) ---", flush=True)
            print(f"Task: {task_prompt[:50]}...", flush=True)
            print(f"Error: {e}", flush=True)
            time.sleep(1)

    result_data["attempts"] = attempts
    return result_data

# --- チャンク処理関数 (変更なし) ---
def process_chunk(chunk, host, chunk_id):
    results = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_task_for_tool_calling, item, host): item for item in chunk}
        with tqdm(total=len(chunk), desc=f"GPU-{chunk_id} Processing", leave=True) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    return results

# --- メイン実行ブロック (printにflush=Trueを追加) ---
def main():
    all_tasks = []
    for task_prompt in tasks:
        base_result_data = {"original_task": task_prompt}
        all_tasks.append({'task_prompt': task_prompt, 'base_result_data': base_result_data})

    if not all_tasks:
        print("❌ Error: The 'tasks' list is empty. Please add tasks to process.", flush=True)
        return

    print(f"Total tasks to process: {len(all_tasks)}", flush=True)

    gpu_ports = [f"127.0.0.1:{BASE_PORT + i}" for i in range(NUM_GPUS)]
    chunk_size = (len(all_tasks) + NUM_GPUS - 1) // NUM_GPUS
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]

    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {executor.submit(process_chunk, chunks[i], gpu_ports[i], i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            all_results.extend(future.result())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    successful_results = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            if result['status'] == 'success':
                formatted_entry = {
                    "messages": [
                        {"role": "user", "content": result['original_task']},
                        {"role": "assistant", "content": result['tool_workflow']}
                    ]
                }
                f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
                successful_results += 1

    print(f"✅ Successfully wrote {successful_results} results to {output_path}", flush=True)

if __name__ == "__main__":
    main()