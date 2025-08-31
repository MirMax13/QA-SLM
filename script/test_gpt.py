import tiktoken
enc = tiktoken.get_encoding("gpt2")

for tok in ["coaching", "swimming", "theories"]:
    print(tok, "->", enc.encode(tok))
def id_to_token(enc, tid: int) -> str:
    return enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace")

for tok_id in [1790, 2646, 3721, 12446, 10946, 7606, 14899, 13101, 9285, 12899, 27982, 41443, 9732]:
    print(tok_id, "->", id_to_token(enc, tok_id))