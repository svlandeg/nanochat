#!/bin/bash

# use previously trained base model
# python -m scripts.base_train --depth=14 --device-batch-size=16 --window-pattern="L"
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run33_d6"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run22_d14"

echo "ll $NANOCHAT_BASE_DIR"
ls -al $NANOCHAT_BASE_DIR

echo "*****************************************************************"

python -m scripts.base_eval --eval "sample"

