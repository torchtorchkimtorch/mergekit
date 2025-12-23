#!/bin/bash
DEVICE=6
CURR_JOB=DARE_4_model_2

OUTPUT_PATH="./merged_models/${CURR_JOB}"

CUDA_VISIBLE_DEVICES=$DEVICE mergekit-yaml yamls/${CURR_JOB}.yaml "$OUTPUT_PATH" --lazy-unpickle --trust-remote-code --cuda --copy-tokenizer

cp /mnt/nlpai-storage/training_team/shared/final_tokenizer/chat_template.jinja "$OUTPUT_PATH/chat_template.jinja"
cp /mnt/nlpai-storage/training_team/shared/final_tokenizer/slow_tokenizer.model "$OUTPUT_PATH/slow_tokenizer.model"
