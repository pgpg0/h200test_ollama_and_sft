import json
import re
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langdetect import detect, LangDetectException

# --- 設定項目 ---
INPUT_DIR = Path('/home/ubuntu/client/Data_azami/input_data/')
OUTPUT_DIR = Path('/home/ubuntu/client/Data_azami/result/multi_result')
MODEL_NAME = "wao/DeepSeek-R1-Distill-Qwen-32B-Japanese"
MAX_ATTEMPTS = 5
NUM_GPUS = 8
BASE_PORT = 11434

# --- LangChain設定 (プロンプトを具体的に変更) ---
PROMPT_TEMPLATE = """
以下の製品情報に基づき、顧客からの具体的な問い合わせとそれに対する返信を、Q&A形式で3セット作成してください。

**必ず以下の指示に従ってください:**

1.  **問い合わせ (Q):**
    * 顧客の会社名（例：「◯◯工業株式会社」「◇◇建設」）と担当者名（例：「◯◯」「山田」）を創作してください。
    * 具体的な工事場所（例：「新宿区」「品川区」）を記載してください。
    * レンタルしたい製品（製品情報にある「商品名」）を、本文中で `**` を使って強調（マークダウンでボールドに）してください。
    * レンタル期間、数量、必要な作業（設置・撤去など）といった具体的な要望を含めてください。
    * 丁寧な依頼文で締めくくってください。

2.  **返信 (A):**
    * 問い合わせてきた顧客の会社名と担当者名を冒頭に記載してください。
    * 問い合わせへの感謝を述べ、内容を承知したことを伝えてください。
    * 製品名を、本文中で `**` を使って強調（マークダウンでボールドに）してください。
    * 見積書の送付や担当者からの連絡など、具体的な次のアクションを提示してください。
    * 丁寧な文章で締めくくってください。

3.  **全体の形式:**
    * 必ず「Q-1.」「A-1.」「Q-2.」「A-2.」「Q-3.」「A-3.」という形式で記述してください。
    * QとAの間、および各セットの間には改行を入れてください。

### 製品情報
{product_info}
"""

# --- ヘルパー関数群 (新しい検証ロジック) ---
def clean_llm_output(text: str) -> str:
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'### (製品情報|回答形式).*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def parse_qa_text(text: str) -> dict | None:
    qa_pairs = {}
    pattern = re.compile(r"(Q-[1-3]\.)\s*(.*?)\s*(A-[1-3]\.)\s*(.*?)(\n(?=Q-[1-3]\.|$)|$)", re.DOTALL)
    matches = pattern.findall(text)
    if len(matches) != 3: return None
    for q_label, q_text, a_label, a_text, _ in matches:
        qa_pairs[q_label] = q_text.strip()
        qa_pairs[a_label] = a_text.strip()
    expected_keys = [f"{p}-{i}." for i in range(1, 4) for p in ("Q", "A")]
    if all(k in qa_pairs for k in expected_keys):
        return qa_pairs
    return None

def is_qa_valid(qa_dict: dict) -> bool:
    if not qa_dict: return False
    try:
        for i in range(1, 4):
            answer = qa_dict.get(f"A-{i}.")
            question = qa_dict.get(f"Q-{i}.")
            if not all([answer, question]): return False
            if detect(answer) != 'ja' or detect(question) != 'ja': return False
            if '**' not in answer or '**' not in question: return False
    except LangDetectException: return False
    return True
    
# ▲▲▲ 新しいプロンプトとヘルパー関数ここまで ▲▲▲

# --- LLMリクエストと検証を行うワーカー関数 ---
def process_product_with_validation(product_item, host):
    product_info_for_prompt = product_item['prompt_input']
    result_data = product_item['base_result_data'].copy()
    result_data.update({"status": "failed", "attempts": 0, "qa_pairs": None, "last_raw_response": None, "original_csv": product_item['original_csv']})
    prompt = PROMPT_TEMPLATE.format(product_info=product_info_for_prompt)
    is_valid = False
    attempts = 0
    while not is_valid and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            response = requests.post(f"http://{host}/api/chat", json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False}, timeout=120)
            response.raise_for_status()
            raw_response = response.json().get("message", {}).get("content", "")
            result_data["last_raw_response"] = raw_response
            parsed_qa = parse_qa_text(clean_llm_output(raw_response))
            if is_qa_valid(parsed_qa):
                is_valid, result_data["status"], result_data["qa_pairs"] = True, "success", parsed_qa
        except Exception as e:
            result_data["last_raw_response"] = f"Error: {e}"
            time.sleep(1)
    result_data["attempts"] = attempts
    return result_data

# --- チャンク処理関数 ---
def process_chunk(chunk, host, chunk_id):
    results = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_product_with_validation, item, host): item for item in chunk}
        with tqdm(total=len(chunk), desc=f"GPU-{chunk_id} Processing", leave=True) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    return results

# --- メイン実行ブロック ---
def main():
    # データ読み込み部分は各フォーマットに合わせて調整
    all_products = []
    csv_files = list(INPUT_DIR.glob('*.csv'))
    for csv_path in csv_files:
        df = pd.read_csv(csv_path).fillna('')
        for _, row in df.iterrows():
            product_name = row['商品名']
            product_info = f"商品名: {row['商品名']}\nカテゴリ: {row['商品カテゴリ']}\n特徴: {row['特徴']}"
            base_result_data = {"product_name": product_name, "category": row['商品カテゴリ'], "features": row['特徴']}
            all_products.append({'prompt_input': product_info, 'base_result_data': base_result_data, 'original_csv': csv_path.name})

    if not all_products: return
    print(f"Format 1: Total products to process: {len(all_products)}")
    
    # 並列処理と結果出力
    gpu_ports = [f"127.0.0.1:{BASE_PORT + i}" for i in range(NUM_GPUS)]
    chunk_size = (len(all_products) + NUM_GPUS - 1) // NUM_GPUS
    chunks = [all_products[i:i + chunk_size] for i in range(0, len(all_products), chunk_size)]
    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {executor.submit(process_chunk, chunks[i], gpu_ports[i], i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            all_results.extend(future.result())

    grouped_results = {}
    for res in all_results:
        grouped_results.setdefault(res.pop('original_csv'), []).append(res)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for original_csv, results_list in grouped_results.items():
        output_path = OUTPUT_DIR / f"{Path(original_csv).stem}_mail.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully wrote {len(results_list)} results to {output_path}")

if __name__ == "__main__":
    main()