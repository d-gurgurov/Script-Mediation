#!/bin/bash
pip install -U vllm
pip install datasets
pip install numpy==1.26.4
pip install soxr

pip install -U accelerate

HF_TOKEN=*
hf auth login --token $HF_TOKEN

MODELS=(llama3-70b llama3-8b aya-8b aya-32b)
LANGS=(ar hi)
 
for model in "${MODELS[@]}"; do
    for lang in "${LANGS[@]}"; do
        echo "=== validate_heads | $model | $lang ==="
        python mech_validate_head.py \
            --model "$model" \
            --lang "$lang" \
            --experiment subsets \
            --k 10 --k-viz 10 \
            --n-prompts 70 \
            --n-bottom-controls 3 --n-random-controls 3 \
            --save-generations --gen-max-new-tokens 50
    done
done