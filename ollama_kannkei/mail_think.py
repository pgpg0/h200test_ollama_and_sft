import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
import subprocess
import sys

# ===================== 設定 =====================
MODEL_PATH = "/data/gpt-oss-20b"  # ローカルにダウンロード済みのモデルパス
INPUT_JSONL_PATH = Path('/home/ubuntu/client/Data_azami/code/chat_sft_data_full.jsonl')
OUTPUT_JSONL_PATH = Path('/home/ubuntu/client/okamura/thinking_results.jsonl')
FAILED_JSONL_PATH = Path('/home/ubuntu/client/okamura/failed_data.jsonl')

BATCH_SIZE = 128            # バッチサイズ（メモリに応じて調整）
MAX_TOKENS = 2048
TEMPERATURE = 0.4
TENSOR_PARALLEL_SIZE = 8  # GPU枚数に合わせて。必要なら変更してください。
UTILIZATION=0.85

# === プロンプト（ユーザー指定のまま） ===
THINKING_PROMPT = """
あなたはB2Bビジネスメールの作成アシスタントです。
受信したメール（question）と、その返信例（answer）が与えられます。
この返信を作成するためにどのような情報整理・判断を行ったのか、
メール作成の思考部分を日本語で論理的に説明してください。

# 制約
- 文章の目的、相手の要望や質問の把握、必要な情報の抽出、トーンや敬語の選択などを含めて説明してください。
- 箇条書きや段階的な説明でも構いません。
- 思考部分の出力のみを生成してください（「thinking:」などの接頭辞は不要）。
- 実際のメール文面は含めないこと。
- 思考部分は以下の形式でお願いします。

【思考部分開始】
思考部分(日本語)
【思考部分終了】

### question
{question}
### answer
{answer}

"""

# ===================== ユーティリティ =====================
def print_gpu_usage():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
        #print("\n=== Current GPU Usage ===")
        #print(result.stdout)
    except Exception:
        pass

def load_input_data(jsonl_file):
    data_pairs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = record.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if user_msg and assistant_msg:
                data_pairs.append({
                    "question": user_msg.get("content", "").strip(),
                    "answer": assistant_msg.get("content", "").strip(),
                    "original_entry": record
                })
    return data_pairs

def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # 先頭に "thinking: " 等があれば除去
    text = re.sub(r'^(thinking|思考|思考プロセス)\s*[:：]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

# ===================== バッチ生成 =====================
def batch_generate(llm: LLM, tasks, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    results = []
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    total = len(tasks)
    pbar = tqdm(total=total, desc="🤔 Generating Thinking", unit="item")

    for i in range(0, total, batch_size):
        batch = tasks[i:i+batch_size]
        prompts = [THINKING_PROMPT.format(question=t['question'], answer=t['answer']) for t in batch]

        try:
            # ここで一括生成（vLLMに推奨される使い方）
            outputs_iter = llm.generate(prompts, sampling_params)
            outputs = list(outputs_iter)  # 安全のため全取得
        except Exception as e:
            # バッチ全体の失敗としてマーク
            for t in batch:
                results.append({
                    'thinking': None,
                    'original_entry': t['original_entry'],
                    'status': f'failed: {e}'
                })
            pbar.update(len(batch))
            continue

        # outputs と batch を対応させる
        for out, t in zip(outputs, batch):
            try:
                # out.outputs は複数のチャンク要素を含むことがあるため連結
                pieces = []
                for step in getattr(out, "outputs", []):
                    # step には text 属性がある想定だが辞書の場合にも対応
                    if hasattr(step, "text"):
                        pieces.append(step.text)
                    elif isinstance(step, dict) and "text" in step:
                        pieces.append(step["text"])
                generated_text = "".join(pieces).strip()
                cleaned = clean_llm_output(generated_text)
                if cleaned:
                    results.append({
                        'thinking': cleaned,
                        'original_entry': t['original_entry'],
                        'status': 'success'
                    })
                else:
                    results.append({
                        'thinking': None,
                        'original_entry': t['original_entry'],
                        'status': 'failed: empty'
                    })
            except Exception as e:
                results.append({
                    'thinking': None,
                    'original_entry': t['original_entry'],
                    'status': f'failed: {e}'
                })

            pbar.update(1)

    pbar.close()
    return results

# ===================== 保存 =====================
def write_results(success_results, failed_results, out_path=OUTPUT_JSONL_PATH, fail_path=FAILED_JSONL_PATH):
    # success_results の original_entry の assistant メッセージを書き換えて保存
    with open(out_path, "w", encoding="utf-8") as f:
        for r in success_results:
            entry = r['original_entry']
            # assistant メッセージを見つけて内容を上書き（堅牢に）
            messages = entry.get("messages", [])
            ass_idx = next((idx for idx,m in enumerate(messages) if m.get("role")=="assistant"), 1 if len(messages)>1 else 0)
            messages[ass_idx]['thinking'] = r['thinking']
            entry['messages'] = messages
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(fail_path, "w", encoding="utf-8") as f:
        for r in failed_results:
            f.write(json.dumps({
                "status": r['status'],
                "original_entry": r['original_entry']
            }, ensure_ascii=False) + "\n")

# ===================== メイン =====================
def main():
    print_gpu_usage()

    print(f"\n🔄 Loading data from {INPUT_JSONL_PATH} ...")
    tasks = load_input_data(INPUT_JSONL_PATH)
    if not tasks:
        print("❌ No valid tasks to process. Exiting.")
        return
    print(f"✅ Loaded {len(tasks)} tasks.")

    print(f"\n🚀 Loading model from {MODEL_PATH} ... (tensor_parallel_size={TENSOR_PARALLEL_SIZE})")
    try:
        llm = LLM(model=MODEL_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE,gpu_memory_utilization=UTILIZATION)
    except Exception as e:
        print(f"Failed to initialize vLLM LLM: {e}", file=sys.stderr)
        raise

    try:
        all_results = batch_generate(llm, tasks, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    finally:
        # 明示的にエンジン終了（vLLMのバージョンによっては shutdown() / close() が無い場合もある）
        try:
            llm.shutdown()
        except Exception:
            pass

    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] != 'success']

    print(f"\n✍️ Writing {len(successful)} successful results to {OUTPUT_JSONL_PATH} ...")
    write_results(successful, failed)

    print("\n--- ✨ Generation Complete! ✨ ---")
    print(f"✔️ Successful: {len(successful)}")
    print(f"❌ Failed:     {len(failed)}")
    print(f"📄 Output file: {OUTPUT_JSONL_PATH}")
    print(f"📄 Failed file: {FAILED_JSONL_PATH}")

if __name__ == "__main__":
    main()
