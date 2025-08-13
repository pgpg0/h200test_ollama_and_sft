import json
import random
import re
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 設定項目 ---
# ★★★ 大量生成用の設定を追加 ★★★
TOTAL_SAMPLES_TO_GENERATE = 1000000 # 生成したいデータの総数
BATCH_SIZE = 10000                  # 1ファイルに保存するデータ数 (1バッチのサイズ)
BASE_FILENAME = "generated_tool_calling_dataset" # 出力ファイル名の共通部分

# --- 通常の設定 ---
OUTPUT_DIR = Path('/home/ubuntu/client/Data_azami/code/synthetic_data_output/fc_data')
MODEL_NAME = "mistral-small"
MAX_ATTEMPTS = 2
NUM_GPUS = 8
BASE_PORT = 11435

# --- テーマリスト ---
TASK_THEMES = [
    # --- 仕事・キャリア関連 ---
    "クライアントとの定例会議の準備",
    "部署内のプロジェクト進捗会議",
    "急な海外出張のフライトとホテルの手配",
    "業界カンファレンスへの参加計画",
    "採用面接のスケジュール調整",
    "出張先でのワーキングスペースの予約",
    "競合他社の新サービスに関する情報収集",
    "チームの歓迎会のお店探しと予約",
    "資格試験の申し込みと勉強計画の立案",
    "社内研修の会場探し",
    "テレワーク日のタスク管理",
    "顧客訪問のための移動ルート確認",
    "チームビルディングイベントの企画",
    "新しいプロジェクトのキックオフミーティング",
    "経費精算のための領収書整理リマインダー",
    "同僚とのランチミーティングの場所探し",
    "キャリアアップのためのセミナー検索",
    "退職する同僚への送別会の準備",
    "オフィスの備品購入リスト作成",
    "年末の業務報告会のスケジュール設定",

    # --- 私生活・自己管理 ---
    "定期的な歯医者の予約",
    "美容院の予約変更",
    "市役所での手続きの必要書類確認",
    "週末の食料品まとめ買いの計画",
    "公共料金の支払いリマインダー設定",
    "パーソナルトレーニングの予約",
    "人間ドックの予約と事前準備の確認",
    "運転免許の更新手続き",
    "お気に入りの服のクリーニング出し",
    "部屋の模様替えのための家具探し",
    "銀行での手続き予約",
    "新しいスマートフォンの機種変更計画",
    "友人への誕生日プレゼント選び",
    "体調不良時の近所の病院探し",
    "フリマアプリへの出品作業",

    # --- 趣味・エンタメ・社交 ---
    "友人とのランチのお店選び",
    "大学時代の同窓会の幹事業務",
    "趣味のカメラサークルの撮影場所探し",
    "好きなアーティストのライブチケット予約",
    "気になる映画の上映時間と映画館のチェック",
    "週末のデートプラン作成",
    "新しくオープンしたカフェ巡り",
    "地域のボランティア活動への参加",
    "習い事（料理教室など）の体験申し込み",
    "スポーツ観戦のチケット手配",
    "カラオケに行く友人との待ち合わせ",
    "ボードゲームカフェの検索と予約",
    "地元の祭りやイベントの情報収集",
    "推し活（イベント参加やグッズ購入）の計画",
    "一人でのんびりできるブックカフェ探し",

    # --- 旅行・観光 ---
    "夏の沖縄旅行の計画",
    "紅葉シーズンの京都への週末旅行",
    "海外旅行のための格安航空券探し",
    "旅行先でのレンタカー予約",
    "日帰りバスツアーの検索と申し込み",
    "キャンプ場の予約と準備",
    "温泉地でのんびりする旅行計画",
    "観光地の穴場スポット探し",
    "パスポートの申請・更新手続き",
    "旅行のお土産リスト作成",
    "グランピング施設の検索と予約",
    "空港までのリムジンバスの時刻表確認",
    "旅行先の天気予報チェック",
    "ご当地グルメのリサーチ",
    "サイクリングロードの検索と計画",

    # --- 家庭・家族サービス ---
    "子供の習い事の送迎スケジュール管理",
    "家族での外食先のレストラン予約",
    "親戚の集まりの場所探しと案内",
    "結婚記念日のサプライズ計画",
    "家電の修理業者への連絡と予約",
    "子供の学校公開日のスケジュール確認",
    "家族でのバーベキュー計画",
    "ペットホテルの検索と予約",
    "子供の誕生日パーティーの準備",
    "地域の公園や遊び場の検索",
    "大型スーパーへの買い出し",
    "マイホーム購入のためのモデルルーム見学予約",
    "両親へのプレゼント選びと配送手配",
    "子供の予防接種のスケジュール管理",
    "不用品の粗大ごみ回収の申し込み",

    # --- 学習・自己啓発 ---
    "大学の友人と図書館での勉強会",
    "オンライン英会話のレッスン予約",
    "興味のある技術系セミナーの検索",
    "資格取得のための通信講座探し",
    "美術館や博物館の特別展のチェック",
    "プログラミングスクールの体験入学",
    "楽器の練習スタジオの予約",
    "読書会の課題図書の購入",
    "講演会のスケジュール確認と申し込み",
    "地域の国際交流イベントへの参加"
]

# --- ツール定義 (省略なし) ---
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

# --- プロンプトテンプレート (省略なし) ---
TASK_GENERATION_PROMPT_TEMPLATE = """
あなたは、ユーザーの多様な要望を想定して作り出すクリエイティブなアシスタントです。
### 指示
1.  以下の「テーマ」に沿った、自然な日本語のユーザーからの「課題」を1つだけ生成してください。
2.  提示された「利用可能なツールリスト」の中から、1つ以上のツールが必要になるような課題を作成してください。
3.  **毎回同じようなシナリオ（例：東京-大阪間の出張）を避け、可能な限り多様な状況を想定してください。**
4.  生成する「課題」のみを出力し、思考プロセスや余計な説明は絶対に含めないでください。
### テーマ
{theme}
### 利用可能なツールリスト
{tool_definitions}
### 良い課題の多様な例
- 今週末、自宅から車で1時間以内で行ける評価4.0以上の温泉宿を3つ提案して。
- 明日の「東京本社での会議」の予定を確認し、会議場所までの自宅からの行き方を教えて。
- 今日の夕方18時に仕事が終わる予定です。その後、渋谷で同僚と会うためのイタリアンレストランを探して予約し、その場所をカレンダーに登録してください。
- カレンダーにある「友人とのランチ」の場所（レストラン名）の詳細情報（正確な住所と電話番号）を調べて。
- 現在地から最も近い映画館を探し、そこまでの徒歩での所要時間を調べて。もし15分以内なら、今夜19時開始の映画の予定をカレンダーに追加して。
### あなたが生成する課題
"""
TOOL_SELECTION_PROMPT_TEMPLATE = """
あなたは優秀なアシスタントです。ユーザーの「課題」を解決するために、どの「ツール」をどのような順番で使うべきか計画を立ててください。
### 課題
{task_prompt}
{tool_definitions}
### 指示
上記の課題を解決するためのワークフローを考え、使用するツール名を順番に提示してください。思考プロセスは不要です。各ツール名は改行して、以下のようにリスト形式で出力してください。
### 回答形式
tool_name1
- parameter1 : ○○
- parameter2 : ○○
tool_name2
- parameter1 : ○○
- parameter2 : ○○
"""

# --- ヘルパー関数 ---
def clean_llm_output(text: str) -> str:
    return text.strip()

def is_tool_list_valid(text: str) -> bool:
    return bool(text and text.strip())

# ★★★ レジューム機能のためのヘルパー関数 ★★★
def get_next_batch_number(output_dir: Path, base_filename: str) -> int:
    """出力ディレクトリを調べて、次のバッチ番号を返す"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.jsonl")
    max_num = 0
    for f in output_dir.glob(f"{base_filename}_*.jsonl"):
        match = pattern.match(f.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1

# --- ワーカー関数 (省略なし) ---
def generate_task_worker(host: str, theme: str):
    prompt_for_llm = TASK_GENERATION_PROMPT_TEMPLATE.format(theme=theme, tool_definitions=TOOL_DEFINITIONS)
    try:
        response = requests.post(f"http://{host}/api/chat", json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_for_llm}], "stream": False}, timeout=180)
        response.raise_for_status()
        generated_task = response.json().get("message", {}).get("content", "")
        cleaned_task = clean_llm_output(generated_task)
        if cleaned_task: return cleaned_task
    except Exception as e:
        print(f"Error during task generation on {host} with theme '{theme}': {e}")
    return None

def process_task_for_tool_calling(task_item, host):
    task_prompt = task_item['task_prompt']
    result_data = task_item['base_result_data'].copy()
    result_data.update({"status": "failed", "attempts": 0, "tool_workflow": None, "last_raw_response": None})
    prompt_for_llm = TOOL_SELECTION_PROMPT_TEMPLATE.format(task_prompt=task_prompt, tool_definitions=TOOL_DEFINITIONS)
    is_valid = False
    attempts = 0
    while not is_valid and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            response = requests.post(f"http://{host}/api/chat", json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_for_llm}], "stream": False}, timeout=300)
            response.raise_for_status()
            raw_response = response.json().get("message", {}).get("content", "")
            result_data["last_raw_response"] = raw_response
            cleaned_response = clean_llm_output(raw_response)
            if is_tool_list_valid(cleaned_response):
                is_valid = True
                result_data["status"] = "success"
                result_data["tool_workflow"] = cleaned_response
        except Exception as e:
            result_data["last_raw_response"] = f"Error: {e}"
            time.sleep(1)
    result_data["attempts"] = attempts
    return result_data

def process_chunk(chunk, host, chunk_id):
    results = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_task_for_tool_calling, item, host): item for item in chunk}
        with tqdm(total=len(chunk), desc=f"GPU-{chunk_id} Workflow Processing", leave=True) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    return results

# ★★★ メイン実行ブロックをバッチ処理対応に変更 ★★★
def main():
    gpu_ports = [f"127.0.0.1:{BASE_PORT + i}" for i in range(NUM_GPUS)]

    start_batch_num = get_next_batch_number(OUTPUT_DIR, BASE_FILENAME)
    total_generated_count = (start_batch_num - 1) * BATCH_SIZE

    print(f"Resuming from batch #{start_batch_num}.")
    print(f"Total samples generated so far: {total_generated_count} / {TOTAL_SAMPLES_TO_GENERATE}")

    current_batch_num = start_batch_num
    while total_generated_count < TOTAL_SAMPLES_TO_GENERATE:
        print(f"\n--- 🚀 Starting Batch #{current_batch_num} ---")
        output_path = OUTPUT_DIR / f"{BASE_FILENAME}_{current_batch_num}.jsonl"

        # --- フェーズ1: 課題の自動生成 ---
        print(f"🤖 Phase 1: Generating {BATCH_SIZE} diverse tasks for batch #{current_batch_num}...")
        generated_tasks = []
        with ThreadPoolExecutor(max_workers=NUM_GPUS * 8) as executor:
            futures = {executor.submit(generate_task_worker, gpu_ports[i % NUM_GPUS], random.choice(TASK_THEMES)): i for i in range(BATCH_SIZE)}
            with tqdm(total=BATCH_SIZE, desc=f"Batch-{current_batch_num} Task Gen") as pbar:
                for future in as_completed(futures):
                    task = future.result()
                    if task: generated_tasks.append(task)
                    pbar.update(1)
        
        if not generated_tasks:
            print(f"❌ Error: No tasks were generated for batch #{current_batch_num}. Skipping.")
            current_batch_num += 1
            continue
        print(f"✅ Phase 1 complete. Successfully generated {len(generated_tasks)} tasks.")

        # --- フェーズ2: 生成された課題に対するツール選択 ---
        print(f"⚙️ Phase 2: Processing {len(generated_tasks)} tasks to create tool workflows...")
        all_tasks_to_process = [{'task_prompt': p, 'base_result_data': {"original_task": p}} for p in generated_tasks]
        chunk_size = (len(all_tasks_to_process) + NUM_GPUS - 1) // NUM_GPUS
        chunks = [all_tasks_to_process[i:i + chunk_size] for i in range(0, len(all_tasks_to_process), chunk_size)]
        all_results = []
        with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            futures = {executor.submit(process_chunk, chunks[i], gpu_ports[i], i): i for i in range(len(chunks))}
            for future in as_completed(futures):
                all_results.extend(future.result())

        # --- フェーズ3: 結果の出力 ---
        successful_results = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for result in all_results:
                if result['status'] == 'success':
                    successful_results += 1
                    formatted_entry = {"messages": [{"role": "user", "content": result['original_task']}, {"role": "assistant", "content": result['tool_workflow']}]}
                    f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
        
        total_generated_count += successful_results
        print(f"--- ✅ Batch #{current_batch_num} complete. Wrote {successful_results} results to {output_path.name} ---")
        print(f"Total progress: {total_generated_count} / {TOTAL_SAMPLES_TO_GENERATE} samples.")
        current_batch_num += 1

    print("\n\n🎉 Target number of samples reached. All batches complete.")

if __name__ == "__main__":
    main()