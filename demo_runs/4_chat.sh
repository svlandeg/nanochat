#!/bin/bash

source .venv/bin/activate

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_run2_base500_sft200"

# chat with your model over a pretty WebUI ChatGPT style
python -m scripts.chat_web
