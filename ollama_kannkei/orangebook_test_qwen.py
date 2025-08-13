import os
import re
import pandas as pd
import json
from pathlib import Path
from langdetect import detect, LangDetectException
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# --- 設定項目 ---
INPUT_DIR = Path('/home/ubuntu/client/Data_azami/input_data/orange_book')
OUTPUT_DIR = Path('/home/ubuntu/client/Data_azami/result')
MAX_ATTEMPTS = 10

# --- LangChain設定 ---
llm = ChatOllama(model="wao/DeepSeek-R1-Distill-Qwen-32B-Japanese")

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。"),
    ("user", """
以下の商品概要を読み、顧客からの問い合わせ例をQ&A形式で3つ作成してください。以下の回答形式に従って作成してください。

### 製品情報
{product_info}

### 回答形式
Q1:
A1:
Q2:
A2:
Q3:
A3:
""")
])

chain = prompt | llm | StrOutputParser()

# --- ヘルパー関数群 (変更なし) ---

def clean_llm_output(text: str) -> str:
    """LLMの出力から余計なヘッダーやマークダウンを削除する"""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'### (製品情報|回答形式).*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def parse_qa_text(text: str) -> dict | None:
    """Q&Aテキストを解析して辞書に変換する"""
    qa_pairs = {}
    pattern = re.compile(r"(Q[1-3]):\s*(.*?)\s*(A[1-3]):\s*(.*?)(\n(?=Q[1-3]|$)|$)", re.DOTALL)
    matches = pattern.findall(text)
    
    if len(matches) != 3:
        return None

    for q_label, q_text, a_label, a_text, _ in matches:
        qa_pairs[q_label] = q_text.strip()
        qa_pairs[a_label] = a_text.strip()
        
    if all(k in qa_pairs for k in ["Q1", "A1", "Q2", "A2", "Q3", "A3"]):
        return qa_pairs
    return None

def is_qa_valid(qa_dict: dict) -> bool:
    """解析されたQ&A辞書が有効か（言語チェックを含む）を検証する"""
    if not qa_dict:
        return False
    try:
        for i in range(1, 4):
            answer = qa_dict.get(f"A{i}")
            if not answer or detect(answer) != 'ja':
                return False
    except LangDetectException:
        return False
    return True

# --- ファイル処理関数 (新しいCSV形式に対応) ---

def generate_qa_from_csv(csv_path: Path, output_json_path: Path):
    """
    単一のCSVファイルを処理し、指定されたパスにJSON結果を出力する関数
    """
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='warn')
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {csv_path}\n{e}")
        return
        
    # ▼▼▼ 変更点: 新しいCSVの必須列をチェック ▼▼▼
    required_columns = ['タイトル', 'clistdisc', 'clistdisc4']
    if not all(col in df.columns for col in required_columns):
        print(f"エラー: {csv_path.name} に必須列（'タイトル', 'clistdisc', 'clistdisc4'）のいずれかが存在しません。")
        return

    all_results = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {csv_path.name}"):
        product_name = row['タイトル'] if pd.notna(row['タイトル']) else "名称不明"
        
        # ▼▼▼ 変更点: 新しい列名から特徴情報を集約 ▼▼▼
        features_list = [
            str(row['clistdisc']) if pd.notna(row['clistdisc']) else '',
            str(row['clistdisc4']) if pd.notna(row['clistdisc4']) else ''
        ]
        product_features = " ".join(filter(None, features_list)).strip()

        product_info = f"商品名: {product_name}\n特徴: {product_features}"

        result_data = {
            "product_name": product_name,
            "source_features": product_features,
            "status": "failed",
            "attempts": 0,
            "qa_pairs": None,
            "last_raw_response": None,
        }
        
        if not product_features:
            result_data["status"] = "skipped (no features)"
            all_results.append(result_data)
            continue

        is_valid = False
        attempts = 0

        while not is_valid and attempts < MAX_ATTEMPTS:
            attempts += 1
            result_data["attempts"] = attempts
            
            try:
                raw_response = chain.invoke({"product_info": product_info})
                result_data["last_raw_response"] = raw_response
                cleaned_text = clean_llm_output(raw_response)
                parsed_qa = parse_qa_text(cleaned_text)
                
                if is_qa_valid(parsed_qa):
                    is_valid = True
                    result_data["status"] = "success"
                    result_data["qa_pairs"] = parsed_qa
                else:
                    if attempts == MAX_ATTEMPTS:
                        print(f"\n   -> ❌ [{product_name}] 検証失敗 (最終試行)")
            except Exception as e:
                print(f"\n   -> ❌ [{product_name}] エラーが発生しました: {e}")
                result_data["last_raw_response"] = str(e)
                break 
        #print(result_data)
        all_results.append(result_data)

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 処理完了。結果を {output_json_path} に保存しました。")
    except Exception as e:
        print(f"\n🚨 JSONファイルへの保存中にエラーが発生しました: {e}")

# --- メイン実行ブロック (変更なし) ---
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(INPUT_DIR.glob('*.csv'))
    
    if not csv_files:
        print(f"入力フォルダ {INPUT_DIR} にCSVファイルが見つかりません。")
    else:
        print(f"{len(csv_files)}個のCSVファイルを処理します。")
        for csv_file_path in csv_files:
            print(f"\n{'='*60}\n処理を開始します: {csv_file_path.name}\n{'='*60}")
            
            output_file_path = OUTPUT_DIR / f"{csv_file_path.stem}_qa_results.json"
            
            generate_qa_from_csv(csv_file_path, output_file_path)
            
        print(f"\n{'='*60}\nすべてのファイルの処理が完了しました。\n{'='*60}")