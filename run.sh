#!/bin/bash

DEVICE=0
OUTPUT_PATH="./merged_models"

CUDA_VISIBLE_DEVICES=$DEVICE mergekit-yaml wbl_ties.yaml "$OUTPUT_PATH" --lazy-unpickle --trust-remote-code --cuda --copy-tokenizer

cp /mnt/nlpai-storage/training_team/shared/final_tokenizer/chat_template.jinja "$OUTPUT_PATH/chat_template.jinja"
