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
MODEL_NAME = "mistral-small"
MAX_ATTEMPTS = 1
NUM_GPUS = 8
BASE_PORT = 11435
OUTPUT_FILENAME = "manual_input_tool_calling.jsonl"

# --- 課題リスト (ここに課題を手入力) ---
tasks = [
    "来週水曜の午後に、新宿駅周辺でクライアントと会うためのカフェを探し、15時から1時間の予定をカレンダーに登録してほしい。",
    "今週末、自宅から車で1時間以内で行ける評価4.0以上の温泉宿を3つ提案して。",
    "明日の「東京本社での会議」の予定を確認し、会議場所までの自宅からの行き方を教えて。",
    "「大阪出張」の予定をカレンダーから検索し、そのホテルの住所を調べて、最寄りのコンビニを3件リストアップしてほしい。",
    "今日の夕方18時に仕事が終わる予定です。その後、渋谷で同僚と会うためのイタリアンレストランを探して予約し、その場所をカレンダーに登録してください。",
    "カレンダーにある「友人とのランチ」の場所（レストラン名）の詳細情報（正確な住所と電話番号）を調べて。",
    "現在地から最も近い映画館を探し、そこまでの徒歩での所要時間を調べて。もし15分以内なら、19時開始の映画の予定をカレンダーに追加して。"
]

# --- ツール定義 (ここに利用させたいツールを記述) ---
TOOL_DEFINITIONS = """
### 利用可能なツールリスト

## Calendar Tools
- tool_name: list-calendars
  description: 利用可能なすべてのカレンダーのリストを取得する
- tool_name: list-events
  description: 指定した期間内のイベントを一覧表示する
- tool_name: search-events
  description: キーワードに一致するイベントを検索する
- tool_name: create-event
  description: 新しいイベントを作成する
- tool_name: update-event
  description: 既存のイベントを更新する
- tool_name: delete-event
  description: 既存のイベントを削除する
- tool_name: get-freebusy
  description: 複数のカレンダーの空き時間情報を確認する

## Google Maps Tools
- tool_name: maps_geocode
  description: 住所を座標（緯度経度）に変換する
- tool_name: maps_reverse_geocode
  description: 座標（緯度経度）を住所に変換する
- tool_name: maps_search_places
  description: テキストクエリを使って場所を検索する
- tool_name: maps_place_details
  description: 特定の場所の詳細情報（連絡先、評価、営業時間など）を取得する
- tool_name: maps_distance_matrix
  description: 複数の地点間の距離と所要時間を計算する
- tool_name: maps_elevation
  description: 地点の標高を調べる
- tool_name: maps_directions
  description: 2地点間のルート案内（道のり、距離、時間）を取得する
"""

# --- プロンプトテンプレート (Tool Calling用に変更) ---
PROMPT_TEMPLATE = """
あなたは優秀なアシスタントです。ユーザーの「課題」を解決するために、どの「ツール」をどのような順番で使うべきか計画を立ててください。

### 課題
{task_prompt}

{tool_definitions}

### 指示
上記の課題を解決するためのワークフローを考え、使用するツール名を順番に提示してください。思考プロセスは不要です。各ツール名は改行して、以下のようにリスト形式で出力してください。

### 回答形式
tool_name_1
tool_name_2
tool_name_3
"""

# --- ヘルパー関数 (変更なし) ---
def clean_llm_output(text: str) -> str:
    return text.strip()

def is_tool_list_valid(text: str) -> bool:
    return bool(text and text.strip())

# --- LLMリクエストと検証を行うワーカー関数 (デバッグプリント追加) ---
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
                timeout=300
            )
            response.raise_for_status()
            raw_response = response.json().get("message", {}).get("content", "")

            # --- ▼▼▼ デバッグ用のprint文を追加 ▼▼▼ ---
            print(f"\n--- Attempt {attempts} on {host} ---")
            print(f"Task: {task_prompt[:50]}...")
            print(f"LLM Raw Response: '{raw_response}'")
            # --- ▲▲▲ デバッグ用のprint文ここまで ▲▲▲ ---

            result_data["last_raw_response"] = raw_response

            cleaned_response = clean_llm_output(raw_response)
            if is_tool_list_valid(cleaned_response):
                is_valid = True
                result_data["status"] = "success"
                result_data["tool_workflow"] = cleaned_response

        except Exception as e:
            result_data["last_raw_response"] = f"Error: {e}"
            # デバッグ用にエラーも表示
            print(f"\n--- ERROR on {host} (Attempt {attempts}) ---")
            print(f"Task: {task_prompt[:50]}...")
            print(f"Error: {e}")
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

# --- メイン実行ブロック (変更なし) ---
def main():
    # 手入力の課題リストから処理対象を作成
    all_tasks = []
    for task_prompt in tasks:
        base_result_data = {"original_task": task_prompt}
        all_tasks.append({'task_prompt': task_prompt, 'base_result_data': base_result_data})

    if not all_tasks:
        print("❌ Error: The 'tasks' list is empty. Please add tasks to process.")
        return

    print(f"Total tasks to process: {len(all_tasks)}")

    # 並列処理
    gpu_ports = [f"127.0.0.1:{BASE_PORT + i}" for i in range(NUM_GPUS)]
    chunk_size = (len(all_tasks) + NUM_GPUS - 1) // NUM_GPUS
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]

    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {executor.submit(process_chunk, chunks[i], gpu_ports[i], i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            all_results.extend(future.result())

    # 結果を単一の.jsonlファイルに出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    successful_results = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            if result['status'] == 'success':
                # SFTに適した形式に変換
                formatted_entry = {
                    "messages": [
                        {"role": "user", "content": result['original_task']},
                        {"role": "assistant", "content": result['tool_workflow']}
                    ]
                }
                f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
                successful_results += 1

    print(f"✅ Successfully wrote {successful_results} results to {output_path}")

if __name__ == "__main__":
    main()