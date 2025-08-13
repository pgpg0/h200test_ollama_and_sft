#!/bin/bash

# このスクリプトは、Ollamaコンテナを起動し、
# 3つの異なるデータ処理スクリプトを並列で実行し、
# 全ての処理が完了した後にコンテナを停止します。
source ../.venv/bin/activate

echo "### STEP 1: Starting all Ollama containers... ###"
# コンテナ起動スクリプトを実行
bash start_containers.sh

# start_containers.sh の終了コードをチェック
if [ $? -ne 0 ]; then
    echo "Error: Failed to start containers. Aborting."
    exit 1
fi

echo -e "\n### STEP 2: Starting all data processing scripts in parallel... ###"

# 3つのPythonスクリプトをバックグラウンドでそれぞれ実行
python /home/ubuntu/client/Data_azami/code/multi_ollama.py 

python /home/ubuntu/client/Data_azami/code/multi_ollama_orange.py 

python /home/ubuntu/client/Data_azami/code/multi_ollama_monotaro.py 

echo -e "\n### STEP 3: Waiting for all processing scripts to complete... ###"
# waitコマンドで、全てのバックグラウンドジョブの完了を待つ
wait
echo "All data processing scripts have finished."

echo -e "\n### STEP 4: Stopping all Ollama containers... ###"
# コンテナ停止スクリプトを実行
bash stop_containers.sh

echo -e "\n### All tasks completed successfully! ###"