#!/bin/bash

source ../../.venv/bin/activate

bash ../stop_containers.sh

bash ../start_containers.sh
sleep 5
#python makedata/ollama_fc_make_test.py
python /home/ubuntu/client/Data_azami/code/makedata/fc_think_test.py
python 
sleep 5
bash ../stop_containers.sh