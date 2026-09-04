#!/bin/bash

pip install -U vllm
pip install datasets
pip install numpy==1.26.4
pip install soxr
pip install fasttext

pip install -U accelerate

HF_TOKEN=*
hf auth login --token $HF_TOKEN

python head_overlap.py

MODELS=(llama3-70b llama3-8b aya-8b aya-32b)
 
for model in "${MODELS[@]}"; do
    echo "=== cross_lingual | $model ==="
    python mech_agnosticism.py --model "$model" --no-pairs
done