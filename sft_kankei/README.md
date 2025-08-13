# SFT実行手順

## 環境構築の躓いた点
https://huggingface.co/datasets/asoria/dataset-notebook-creator-content/blob/main/NousResearch-hermes-function-calling-v1-sft-16db5c5e-1dcb-4532-aecf-36b50a086d59.ipynb
こちらを元に開始
ただし、ライブラリの更新関係でそのままでは動かず大きく変更(具体的には、argsをSFTConfigにし、Trainerの中の引数などをSFTConfigに移動)
また、formatting_funcでデータセットに対して実行できるよう調整

## 実行方法
1. source /home/ubuntu/client/Data_azami/code/sft/sft_test/bin/activate 
2. python {sft_script}

モデルや使用するデータセットの変更などはスクリプト内を変更して実行
NousResearch/hermes-function-calling-v1は
dataset = load_dataset("NousResearch/hermes-function-calling-v1", name="func_calling", split="train")
のnameのとこを変更することでsingle_turnとmulti_turnの切り替え可能(ただし、中身のデータ形式が違うため、
訓練データの整形が必要)
# SFT用スクリプト
/home/ubuntu/client/Data_azami/code/sft/sft_test_qwen.py : QwenモデルをNousResearch/hermes-function-calling-v1を用いてSFTを行うスクリプト(single turn)

/home/ubuntu/client/Data_azami/code/sft/sft_test_lora.py：QwenモデルをNousResearch/hermes-function-calling-v1を用いてSFTをLoRAで行うスクリプト(single turn)

/home/ubuntu/client/Data_azami/code/sft/sft_test_qwen_multi.py：QwenモデルをNousResearch/hermes-function-calling-v1を用いてSFTを行うスクリプト(multi turn)

# テスト用スクリプト
/home/ubuntu/client/Data_azami/code/sft/inference_sft.py：Qwen/Qwen3-32BとSFT後の推論能力をテストする。訓練時にevalデータとしていたものからランダムに3件選んでテストを行う

/home/ubuntu/client/Data_azami/code/sft/aimai_test.py：一つのテストデータのuser_promptを三段階の曖昧さに分けてテストする。

/home/ubuntu/client/Data_azami/code/sft/gorilla/berkeley-function-call-leaderboard/leaderboard_test.sh：ベンチマークテスト用 