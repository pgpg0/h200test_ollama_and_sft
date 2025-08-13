import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ollamaへの接続設定
llm = ChatOllama(model="deepseek-r1:70b")

# 通常のプロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。"),
    ("user", "{question}")
])

# 出力から<think>...</think>タグを削除する関数
def remove_think_tags(text):
    # <think>から</think>までを、改行を含めて全てマッチさせて空文字に置換
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

# LangChainのチェイン
chain = prompt | llm | StrOutputParser()

# 質問を実行
question = """
以下の商品概要を読み、顧客からの問い合わせ例をQ&A形式で作成してください。
### 製品情報
**製品名:**
モノタロウ プレセット型トルクレンチ (ケース付)

**特徴:**
* **設定トルクが一目でわかる:** 本体に刻まれた目盛りによって、締め付けたいトルク値をあらかじめ設定できます。
* **締め付け完了を音と感触でお知らせ:** 設定したトルクに達すると「カチッ」という音と軽いショックで知らせてくれるため、オーバートルク（締めすぎ）を防ぎます。
* **幅広いトルク範囲:** 小ねじから自動車のホイールナットのような大径ボルトまで、様々な製品ラインナップから作業に応じたトルク範囲のレンチを選ぶことができます。
* **右回転方向のみ:** 締め付け専用（右回転）のトルクレンチです。緩める作業には使用できません。
* **専用ケース付き:** 保管や持ち運びに便利なブローケースが付属しています。

**値段:**
モデルやトルク範囲によって価格は異なりますが、一般的な乗用車のタイヤ交換に使用できるモデル（例：トルク調整範囲 40～200 N・m、差込角12.7mm）は、**4,728円（税込）**から販売されています（2025年7月時点）。

**主な用途:**
* 自動車やバイクのホイールナットの締め付け
* エンジン回りのボルト・ナットの締め付け
* 各種機械の組み立て・メンテナンス

**ご注意:**
* トルクレンチは精密測定工具です。落下させたり、強い衝撃を与えたりしないでください。
* 使用後は、トルク設定を最低値に戻して保管することが推奨されています。
"""
raw_response = chain.invoke({"question": question})

# 思考タグを削除してクリーンな応答を取得
clean_response = remove_think_tags(raw_response)

# print("--- RAW Response (思考あり) ---")
# print(raw_response)
print("\n--- Clean Response (思考なし) ---")
print(clean_response)
