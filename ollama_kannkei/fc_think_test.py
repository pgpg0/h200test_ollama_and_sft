import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
import time

# --- 設定項目 ---
# ★ 入力と出力のファイルパスを指定
INPUT_JSONL_PATH = Path('/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data_format/converted_truly_final_1_namiuchi_ver2.jsonl')
OUTPUT_JSONL_PATH = Path('/home/ubuntu/client/Data_azami/code/synthetic_data_output/with_think_data/generated_thinking_new_format.jsonl')

# ★ モデルとAPI設定
MODEL_NAME = "mistral-small"
MAX_ATTEMPTS = 3
NUM_GPUS = 8
BASE_PORT = 11435

# --- プロンプトテンプレート ---
# tool_calls形式に対応したプロンプト
THINKING_GENERATION_PROMPT_TEMPLATE = """
ユーザーの質問（question）と、その質問に回答するためのツール呼び出しリスト（answer）が提示されます。
ユーザーの質問を達成するために、なぜそのツール呼び出し（answer）が必要なのか、その順序で呼び出す必要があるのかを論理的に説明する思考プロセス（thinking）を生成してください。

# 制約
- 各ツールの役割や目的を中心に、ステップバイステップで説明してください。
- 必ず日本語で、自然で分かりやすい文章で記述してください。
- thinkingの出力のみを生成してください。余計な接頭辞（「thinking:」など）は含めないでください。

# fewshot例
## 例1
### question
明日の午後3時に大型スーパーへの買い出しに行く予定です。家から最も近い大型スーパーを検索し、その場所までの道順と所要時間を教えてください。また、そのスーパーの営業時間を確認して、3時に行けるかどうかをチェックしてください。
### answer
[
  {{
    "name": "maps_search_places",
    "arguments": "{{\\"query\\": \\"大型スーパー\\"}}"
  }},
  {{
    "name": "maps_place_details",
    "arguments": "{{\\"place_id\\": \\"maps_search_placesの結果から取得\\"}}"
  }},
  {{
    "name": "maps_directions",
    "arguments": "{{\\"origin\\": \\"家の住所\\", \\"destination\\": \\"maps_place_detailsの結果から取得\\"}}"
  }}
]
### thinking
ユーザーは特定の日時に買い物に行くための計画を立てたいと考えている。これを実現するには、以下の情報を段階的に取得する必要がある。
1. **場所の検索:** まず、「大型スーパー」というキーワードで自宅に最も近い場所を検索する必要がある。`maps_search_places`ツールを使用する。
2. **詳細情報の取得:** 次に、見つかったスーパーが「午後3時」に営業しているか確認する必要がある。`maps_place_details`ツールを使って、場所のID（place_id）から営業時間などの詳細情報を取得する。
3. **ルートの確認:** 最後に、自宅からそのスーパーまでの具体的な道順と所要時間を調べる。`maps_directions`ツールで、自宅からスーパーまでのルートを検索する。
これらのステップを順に実行することで、ユーザーの買い物計画をサポートできる。

# 本番
### question
{question}
### answer
{answer}
### thinking
"""

# --- ヘルパー関数 ---

def load_input_data(path: Path) -> list[dict]:
    """
    新しい形式のJSONLファイルを読み込み、処理用データに変換する。
    """
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    messages = entry.get('messages', [])
                    if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                        user_msg = messages[0]
                        assistant_msg = messages[1]

                        question = user_msg.get('content')
                        tool_calls = assistant_msg.get('tool_calls')

                        if question and isinstance(tool_calls, list):
                            # プロンプト用に tool_calls をインデント付きのJSON文字列に変換
                            answer_str = json.dumps(tool_calls, ensure_ascii=False, indent=2)
                            data.append({
                                'question': question,
                                'answer_str': answer_str, # プロンプト用の文字列
                                'original_entry': entry   # 元のデータ全体を保持
                            })
                except (json.JSONDecodeError, KeyError, IndexError):
                    print(f"Warning: Skipping malformed or unexpected JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {path}")
        return []
    return data

def clean_llm_output(text: str) -> str:
    """LLMの出力から不要な空白や接頭辞を削除する"""
    text = text.strip()
    text = re.sub(r'^(thinking|思考|思考プロセス)\s*[:：]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


# --- ワーカー関数 ---

def generate_thinking_worker(item: dict, host: str):
    """
    1つのデータ項目からthinkingを生成するワーカー。
    """
    prompt_for_llm = THINKING_GENERATION_PROMPT_TEMPLATE.format(
        question=item['question'],
        answer=item['answer_str']
    )

    attempts = 0
    while attempts < MAX_ATTEMPTS:
        try:
            response = requests.post(
                f"http://{host}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt_for_llm}],
                    "stream": False
                },
                timeout=240
            )
            response.raise_for_status()
            generated_text = response.json().get("message", {}).get("content", "")
            cleaned_thinking = clean_llm_output(generated_text)

            if cleaned_thinking:
                # 成功したら、生成した思考と元のデータを返す
                return {
                    'thinking': cleaned_thinking,
                    'original_entry': item['original_entry'],
                    'status': 'success'
                }
            else:
                attempts += 1
                print(f"Warning: Empty response from {host}. Retrying ({attempts}/{MAX_ATTEMPTS})...")
                time.sleep(2)

        except requests.exceptions.RequestException as e:
            attempts += 1
            print(f"Error on {host}: {e}. Retrying ({attempts}/{MAX_ATTEMPTS})...")
            time.sleep(5)

    # 全てのリトライが失敗した場合
    return {
        'thinking': None,
        'original_entry': item['original_entry'],
        'status': 'failed'
    }


# --- メイン実行ブロック ---

def main():
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    gpu_ports = [f"127.0.0.1:{BASE_PORT + i}" for i in range(NUM_GPUS)]

    print(f"🔄 Loading data from {INPUT_JSONL_PATH}...")
    tasks_to_process = load_input_data(INPUT_JSONL_PATH)
    if not tasks_to_process:
        print("❌ No valid tasks to process. Exiting.")
        return
    print(f"✅ Loaded {len(tasks_to_process)} tasks.")

    print(f"🤖 Starting 'thinking' generation for {len(tasks_to_process)} tasks using {NUM_GPUS} workers...")
    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS * 16) as executor:
        futures = {
            executor.submit(generate_thinking_worker, task, gpu_ports[i % NUM_GPUS]): task
            for i, task in enumerate(tasks_to_process)
        }
        with tqdm(total=len(tasks_to_process), desc="🤔 Generating Thinking") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                pbar.update(1)

    # ★★★【変更箇所】★★★
    # 3. 結果の集計と、指定された構造での保存
    successful_results = [res for res in all_results if res['status'] == 'success']
    failed_count = len(all_results) - len(successful_results)

    print(f"\n✍️ Writing {len(successful_results)} successful results to {OUTPUT_JSONL_PATH}...")
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for result in successful_results:
            # 元のデータを取得
            entry_to_write = result['original_entry']
            
            # assistantメッセージのcontentを生成したthinkingで更新
            # entry_to_write['messages'][1]がassistantメッセージであると想定
            assistant_message = entry_to_write['messages'][1]
            assistant_message['content'] = result['thinking']
            
            f.write(json.dumps(entry_to_write, ensure_ascii=False) + '\n')

    print("\n--- ✨ Generation Complete! ✨ ---")
    print(f"✔️ Successful: {len(successful_results)}")
    print(f"❌ Failed:     {failed_count}")
    print(f"📄 Output file: {OUTPUT_JSONL_PATH}")
    print("🎉 All done.")

if __name__ == "__main__":
    main()