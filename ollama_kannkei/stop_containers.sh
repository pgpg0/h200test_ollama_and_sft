#!/bin/bash

# 使用したGPUの数を自動で検出
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Stopping and removing ${NUM_GPUS} Ollama containers..."

for i in $(seq 0 $((NUM_GPUS - 1)))
do
  CONTAINER_NAME="ollama-gpu${i}"
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}"
done

echo -e "\nAll specified containers have been stopped."