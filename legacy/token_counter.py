
import json
import nltk
nltk.download('punkt_tab')
dataset_path = "datasets/ChatGPT/generative/fridge_dataset_v1.3_clean.json"

with open(dataset_path, "r") as f:
    dataset = json.load(f)


token_count = 0
pairs = [(item["instruction"], item["response"]) for item in dataset]

all_text = " ".join([text for pair in pairs for text in pair])
tokens = nltk.word_tokenize(all_text)

token_count = len(tokens)


#Unique tokens
unique_tokens = set(tokens)

print(f"Total number of unique tokens: {len(unique_tokens)}")

import json
import tiktoken
from collections import Counter

dataset_path = "datasets/ChatGPT/generative/fridge_dataset_v1.3_clean.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Використовуємо GPT-2 токенізатор
enc = tiktoken.get_encoding("gpt2")

# Збираємо всі токени
all_tokens = []

for item in dataset:
    # Об'єднуємо instruction та response
    combined_text = item["instruction"] + " " + item["response"]
    
    # Токенізуємо
    tokens = enc.encode(combined_text)
    all_tokens.extend(tokens)

# Підраховуємо унікальні токени
unique_tokens = set(all_tokens)
token_counts = Counter(all_tokens)

print(f"Кількість унікальних токенів: {len(unique_tokens):,}")

from typing import List, Iterable, Counter
import tiktoken
import json

def read_json_array_records(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {json_path}")
    return data

def load_texts_fridge_json(json_path: str) -> List[str]:
    records = read_json_array_records(json_path)
    texts: List[str] = []
    for rec in records:
        instr = rec.get("instruction") or ""
        resp = rec.get("response") or ""
        joined = (str(instr).strip() + "\n" + str(resp).strip()).strip()
        if joined:
            texts.append(joined)
    return texts

a_text = load_texts_fridge_json("datasets/ChatGPT/generative/fridge_dataset_v1.3_clean.json")
def tokenize_bpe_gpt2(texts: Iterable[str]) -> List[str]:
    enc = tiktoken.get_encoding("gpt2")
    tokens: List[str] = []
    for t in texts:
        ids = enc.encode(t, disallowed_special=())
        tokens.extend(enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids)
    return tokens

a_bpe = tokenize_bpe_gpt2(a_text)


def compute_counter(tokens: Iterable[str]) -> Counter:
    return Counter(tokens)


a_bpe_counter = compute_counter(a_bpe)

print(f"Unique tokens count: {len(a_bpe_counter)}")