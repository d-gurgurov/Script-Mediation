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
    python reps_vectors.py --model "$model" --pooling last_token_instruct
done