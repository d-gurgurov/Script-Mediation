#!/bin/bash

pip install -U vllm
pip install datasets
pip install numpy==1.26.4
pip install soxr

pip install -U accelerate

HF_TOKEN=*
hf auth login --token $HF_TOKEN

MODELS=(llama3-70b llama3-8b aya-8b aya-32b)
LANGS=(hi ar)
 
for model in "${MODELS[@]}"; do
    for lang in "${LANGS[@]}"; do
        echo "=== heads | $model | $lang ==="
        python mech_localize.py --model "$model" --lang "$lang" --experiment patch
    done
done