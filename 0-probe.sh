#!/bin/bash

pip install -U vllm
pip install datasets
pip install numpy==1.26.4
pip install soxr

pip install -U accelerate

HF_TOKEN=*
hf auth login --token $HF_TOKEN

MODELS=(llama3-70b llama3-8b aya-8b aya-32b)
 
for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    python reps_probes.py --model "$model" --lang "hi"
    python reps_similarity.py --model "$model"
done