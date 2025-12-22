#!/bin/bash

# 출력 경로 설정
OUTPUT_PATH="./merged_models/tie-merged"

# mergekit 실행
mergekit-yaml wbl_ties.yaml "$OUTPUT_PATH" --lazy-unpickle --trust-remote-code --cuda --copy-tokenizer

cp /mnt/nlpai-storage/training_team/shared/final_tokenizer/chat_template.jinja "$OUTPUT_PATH/chat_template.jinja"
