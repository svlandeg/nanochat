import os, torch
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
special_ids = {tokenizer.encode_special(t) for t in tokenizer.get_special_tokens()}
token_bytes = [
    0 if i in special_ids else len(tokenizer.enc.decode_single_token_bytes(i))
    for i in range(vocab_size)
]
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes, path)
print(f"Saved {path}")
