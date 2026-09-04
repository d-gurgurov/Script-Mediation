#!/bin/bash

pip install fasttext

HF_TOKEN=*
hf auth login --token $HF_TOKEN

for model in llama3-70b llama3-8b aya-8b aya-32b; do
  for lang in ar hi; do
    for source in patch_abs; do
      run "analyze_${model}_${lang}_${source}" \
        python mech_success_head.py \
          --model "$model" --lang "$lang" --source "$source" \
          --targets single subsets
    done
  done
done

