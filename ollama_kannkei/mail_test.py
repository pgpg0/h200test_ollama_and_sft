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
INPUT_DIR = Path('/home/ubuntu/client/Data_azami/input_data')
OUTPUT_DIR = Path('/home/ubuntu/client/Data_azami/result')
MAX_ATTEMPTS = 10

# --- LangChain設定 (プロンプトを具体的に変更) ---
llm = ChatOllama(model="wao/DeepSeek-R1-Distill-Qwen-32B-Japanese")

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは、建設機械レンタル会社の営業担当者です。提供された製品情報に基づき、顧客からの具体的な問い合わせメールと、それに対する丁寧な返信メールを作成するプロのライターです。"),
    ("user", """
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

### 回答形式の例
Q-1.
◯◯工業株式会社の◯◯と申します。新宿区で改修工事を行っており、資材の搬送および作業員の昇降用に**仮設昇降機（工事用エレベーター）**のレンタルを希望しています。6階建ての建物に対応できる昇降機1台を2ヶ月間レンタルした場合の見積もり（設置・撤去費用含む）をお願いいたします。安全点検や操作指導の可否についても教えていただけると助かります。よろしくお願い致します。
A-1.
◯◯工業株式会社　◯◯様
平素よりお世話になっております。ご連絡ありがとうございます。**仮設昇降機（工事用エレベーター）**の件、6階建て建物対応の機種をご希望とのこと、承知いたしました。当社には各種昇降機がございますので、ご要望に応じて最適な機種をご提案いたします。見積書につきましては、お問い合わせ内容をもとに作成し後日メールにて送付いたします。また、安全点検や操作指導については対応可能ですので、必要に応じて現場講習を行わせていただきます。何かご質問がありましたらお知らせください。
""")
])


chain = prompt | llm | StrOutputParser()

# --- ヘルパー関数群 (一部変更) ---

def clean_llm_output(text: str) -> str:
    """LLMの出力から余計なヘッダーやマークダウンを削除する"""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 削除パターンから "回答形式の例" を削除し、意図しない削除を防ぐ
    text = re.sub(r'### (製品情報|回答形式).*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def parse_qa_text(text: str) -> dict | None:
    """Q&Aテキストを解析して辞書に変換する"""
    qa_pairs = {}
    # 正規表現を 'Q-1.' 'A-1.' の形式に更新
    pattern = re.compile(r"(Q-[1-3]\.)\s*(.*?)\s*(A-[1-3]\.)\s*(.*?)(\n(?=Q-[1-3]\.|$)|$)", re.DOTALL)
    matches = pattern.findall(text)
    
    if len(matches) != 3:
        return None

    for q_label, q_text, a_label, a_text, _ in matches:
        qa_pairs[q_label] = q_text.strip()
        qa_pairs[a_label] = a_text.strip()
        
    # チェックするキーを 'Q-1.', 'A-1.' などに更新
    expected_keys = [f"{prefix}-{i}." for i in range(1, 4) for prefix in ("Q", "A")]
    if all(k in qa_pairs for k in expected_keys):
        return qa_pairs
    return None

def is_qa_valid(qa_dict: dict) -> bool:
    """解析されたQ&A辞書が有効か（言語チェック・形式チェックを含む）を検証する"""
    if not qa_dict:
        return False
    try:
        for i in range(1, 4):
            # 辞書から取得するキーを 'A-1.' の形式に更新
            answer_key = f"A-{i}."
            question_key = f"Q-{i}."
            answer = qa_dict.get(answer_key)
            question = qa_dict.get(question_key)

            # 回答と質問が空でないか、日本語か、** が含まれているかをチェック
            if not all([answer, question]):
                return False
            if detect(answer) != 'ja' or detect(question) != 'ja':
                return False
            if '**' not in answer or '**' not in question:
                return False
                
    except LangDetectException:
        return False
    return True

# --- ファイル処理関数 (変更なし) ---

def generate_qa_from_csv(csv_path: Path, output_json_path: Path):
    """
    単一のCSVファイルを処理し、指定されたパスにJSON結果を出力する関数
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        return

    all_results = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {csv_path.name}"):
        product_name = row['商品名']
        product_info = f"商品名: {row['商品名']}\nカテゴリ: {row['商品カテゴリ']}\n特徴: {row['特徴']}"
        
        result_data = {
            "product_name": product_name,
            "category": row['商品カテゴリ'],
            "features": row['特徴'],
            "status": "failed",
            "attempts": 0,
            "qa_pairs": None,
            "last_raw_response": None,
        }
        
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
        print(result_data)
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
            
            output_file_path = OUTPUT_DIR / f"{csv_file_path.stem}_mail_results.json"
            
            generate_qa_from_csv(csv_file_path, output_file_path)
            
        print(f"\n{'='*60}\nすべてのファイルの処理が完了しました。\n{'='*60}")