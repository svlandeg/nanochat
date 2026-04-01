#!/bin/bash

source .venv/bin/activate

# Use previously tuned model
# python -m scripts.chat_sft --eval-every=-1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run2_base500_sft200"

echo "ll $NANOCHAT_BASE_DIR"
ls -al $NANOCHAT_BASE_DIR

# python -m scripts.chat_eval -i sft

echo "*****************************************************************"

# chat with the model over CLI! 
python -m scripts.chat_cli -p "What is the capital of France?"
# python -m scripts.chat_cli

