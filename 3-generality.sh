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
    echo "=== generality | $model ==="
    python reps_generality.py --model "$model" --all-layers
done

python reps_plot_generality.py --model llama3-8b --scale 0.25

python reps_plot_generality.py --model llama3-70b --scale 0.1

python reps_plot_generality.py --model aya-8b --scale 1.0

python reps_plot_generality.py --model aya-32b --scale 1.0