import json
from collections import Counter
from typing import List
import tiktoken
from tqdm.auto import tqdm
import time
import os

FRIDGE_JSON = "datasets/ChatGpt/generative/fridge_dataset_v1.3_clean.json"
TINYSTORIES_TRAIN_JSON = "filtered_tinystories/tiny_stories_train_filtered_3.json"
TINYSTORIES_VAL_JSON = "filtered_tinystories/tiny_stories_val_filtered_3.json"

OUT_DIR = "filtered_tinystories"
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

def load_texts_tinystories_json_lines(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return [obj["text"] for obj in texts]

def load_texts_fridge_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [row["instruction"] + " " + row["response"] for row in data]

def compute_token_ids(texts: List[str]) -> Counter:
    counter = Counter()
    for text in tqdm(texts, desc="Counting tokens"):
        counter.update(enc.encode(text, disallowed_special=()))
    return counter

def filter_texts(texts: List[str], tokens_to_delete: set) -> List[str]:
    filtered = []
    for text in tqdm(texts, desc="Filtering stories"):
        token_ids = enc.encode(text, disallowed_special=())
        if not any(tid in tokens_to_delete for tid in token_ids):
            filtered.append(text)
    return filtered

def save_texts_as_json(texts: List[str], out_path: str):
    data = [{"text": t} for t in texts]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("Loading datasets...")
fridge_texts = load_texts_fridge_json(FRIDGE_JSON)
tiny_train = load_texts_tinystories_json_lines(TINYSTORIES_TRAIN_JSON)
tiny_val = load_texts_tinystories_json_lines(TINYSTORIES_VAL_JSON)

print("Counting tokens in fridge dataset...")
fridge_token_ids = compute_token_ids(fridge_texts)

print("Counting tokens in TinyStories train...")
tinystories_train_token_ids = compute_token_ids(tiny_train)

print("Counting tokens in TinyStories val...")
tinystories_val_token_ids = compute_token_ids(tiny_val)

token_rare = 5
rare_train_tokens = {tid for tid, c in tinystories_train_token_ids.items() if c <= token_rare}
rare_val_tokens = {tid for tid, c in tinystories_val_token_ids.items() if c <= token_rare}
fridge_tokens_set = set(fridge_token_ids.keys())

tokens_to_delete_train = rare_train_tokens - fridge_tokens_set
tokens_to_delete_val = rare_val_tokens - fridge_tokens_set

print(f"Found {len(tokens_to_delete_train)} rare tokens to delete in train")
print(f"Found {len(tokens_to_delete_val)} rare tokens to delete in val")

all_tokens_before = set(fridge_token_ids.keys()) | set(tinystories_train_token_ids.keys()) | set(tinystories_val_token_ids.keys())
print(f"Unique tokens before filtering: {len(all_tokens_before)}")

filtered_train = filter_texts(tiny_train, tokens_to_delete_train)
filtered_val = filter_texts(tiny_val, tokens_to_delete_val)

print(f"Original train stories: {len(tiny_train)}, filtered: {len(filtered_train)}")
print(f"Original val stories: {len(tiny_val)}, filtered: {len(filtered_val)}")

# -----------------------------
# Статистика після обрізання
# -----------------------------
# train_tokens_after = compute_token_ids(filtered_train)
# val_tokens_after = compute_token_ids(filtered_val)
# all_tokens_after = set(fridge_token_ids.keys()) | set(train_tokens_after.keys()) | set(val_tokens_after.keys())
# print(f"Unique tokens after filtering: {len(all_tokens_after)}")
# print(f"Tokens removed from vocabulary: {len(all_tokens_before) - len(all_tokens_after)}")

save_texts_as_json(filtered_train, os.path.join(OUT_DIR, f"tiny_stories_train_filtered_{token_rare}.json"))
save_texts_as_json(filtered_val, os.path.join(OUT_DIR, f"tiny_stories_val_filtered_{token_rare}.json"))
print(f"Filtered datasets saved in {OUT_DIR}")
