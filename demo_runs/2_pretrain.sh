#!/bin/bash

source .venv/bin/activate

# use previously trained base model
# python -m scripts.base_train --depth=6 --device-batch-size=16 --window-pattern="L"
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run1_base50"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run2_base500"

echo "ll $NANOCHAT_BASE_DIR"
ls -al $NANOCHAT_BASE_DIR

echo "*****************************************************************"

python -m scripts.base_eval --eval "sample"

