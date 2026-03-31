#!/bin/bash

# Use previously tuned model
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# python -m scripts.chat_sft --eval-every=-1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run22_d14_sft"

echo "ll $NANOCHAT_BASE_DIR"
ls -al $NANOCHAT_BASE_DIR

# python -m scripts.chat_eval -i sft

echo "*****************************************************************"

# chat with the model over CLI! 
# python -m scripts.chat_cli -p "What is the capital of France?"
python -m scripts.chat_cli

