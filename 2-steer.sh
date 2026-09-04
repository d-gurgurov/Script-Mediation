#!/bin/bash

pip install -U vllm
pip install datasets
pip install numpy==1.26.4
pip install soxr
pip install fasttext

pip install -U accelerate

HF_TOKEN=*
hf auth login --token $HF_TOKEN

MODELS=(llama3-8b aya-8b llama3-70b aya-32b)
LANGS=(hi ar)
 
for model in "${MODELS[@]}"; do
    for lang in "${LANGS[@]}"; do
        echo "=== $model | $lang ==="
        python reps_intervene.py --model "$model" --lang "$lang" --all-layers
        python reps_success.py --model "$model" --lang "$lang" --all-layers
    done
done

python reps_judge.py --input_dir results --output_dir results_quality