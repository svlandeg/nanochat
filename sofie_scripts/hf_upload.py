from huggingface_hub import login, upload_file


login()


upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/base_checkpoints/d14/meta_002192.json", path_in_repo="base/meta_002192.json", repo_id="Sofie/nanochat_d14", repo_type="model")
upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/base_checkpoints/d14/model_002192.pt", path_in_repo="base/model_002192.pt", repo_id="Sofie/nanochat_d14", repo_type="model")

upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/chatsft_checkpoints/d14/meta_000971.json", path_in_repo="sft/meta_000971.json", repo_id="Sofie/nanochat_d14", repo_type="model")
upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/chatsft_checkpoints/d14/model_000971.pt", path_in_repo="sft/model_000971.pt", repo_id="Sofie/nanochat_d14", repo_type="model")

upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/tokenizer/token_bytes.pt", path_in_repo="token_bytes.pt", repo_id="Sofie/nanochat_d14", repo_type="model")
upload_file(path_or_fileobj="/home/ubuntu/.cache/nanochat_run22_d14_sft/tokenizer/tokenizer.pkl", path_in_repo="tokenizer.pkl", repo_id="Sofie/nanochat_d14", repo_type="model")
