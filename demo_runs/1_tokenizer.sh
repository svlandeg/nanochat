#!/bin/bash

# Default intermediate artifacts directory is in ~/.cache/nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# install the repo dependencies
uv sync --extra cpu
source .venv/bin/activate

# Download some pretraining data
python -m nanochat.dataset -n 8

echo "*****************************************************************"

# train and evaluate the tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

echo "*****************************************************************"

echo "ll ~/.cache/nanochat"
ls -l $NANOCHAT_BASE_DIR
