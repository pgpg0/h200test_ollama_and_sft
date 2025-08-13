#!/bin/bash

# --- 設定 ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
BASE_PORT=11437
MODEL_NAME="gpt-oss:20b"
IMAGE_NAME="ollama/ollama"
SHARED_MODELS_DIR="/data/ollama-model-shared"
NUM_PARALLEL=128 # OLLAMA_NUM_PARALLELの値を設定

# --- 準備 ---
mkdir -p "${SHARED_MODELS_DIR}"
echo "Models will be shared in: ${SHARED_MODELS_DIR}"
echo -e "\nDetected ${NUM_GPUS} GPUs. Starting containers..."

# --- コンテナ起動 ---
for i in $(seq 0 $((NUM_GPUS - 1)))
do
  HOST_PORT=$((BASE_PORT + i))
  CONTAINER_NAME="ollama-gpu${i}"

  echo "Starting ${CONTAINER_NAME} on GPU ${i}, mapping to host port ${HOST_PORT}..."

  docker run -d --rm \
    --gpus "device=${i}" \
    -v "${SHARED_MODELS_DIR}:/root/.ollama" \
    -p "${HOST_PORT}:11434" \
    --name "${CONTAINER_NAME}" \
    -e OLLAMA_NUM_PARALLEL=${NUM_PARALLEL} \
    -e OLLAMA_FLASH_ATTENTION=1 \
    "${IMAGE_NAME}"
done

echo -e "\nWaiting for containers to initialize..."
sleep 5



# # --- モデルのプル ---
# echo -e "\nPulling model [${MODEL_NAME}] using representative container (ollama-gpu0)..."
# docker exec ollama-gpu0 ollama pull "${MODEL_NAME}"

echo -e "\nLoading model into each container..."

for i in $(seq 0 $((NUM_GPUS - 1)))
do
  CONTAINER_NAME="ollama-gpu${i}"
  echo "Loading model in ${CONTAINER_NAME}..."
  docker exec "${CONTAINER_NAME}" ollama pull "${MODEL_NAME}" &
done

wait
echo "All models loaded."

echo -e "\nAll containers are ready. Each is configured with OLLAMA_NUM_PARALLEL=${NUM_PARALLEL}."