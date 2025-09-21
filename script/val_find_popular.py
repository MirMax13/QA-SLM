import json
import os
import random
from collections import Counter
from typing import List, Dict, Tuple

import tiktoken
from tqdm.auto import tqdm

FRIDGE_JSON = "datasets/ChatGpt/generative/fridge_dataset_v1.3_clean.json"
TINYSTORIES_TRAIN_JSON = "filtered_tinystories/tiny_stories_train_filtered_5.json"  
TINYSTORIES_VAL_JSON   = "filtered_tinystories/tiny_stories_val_filtered_5.json"

OUT_DIR = "eval_selection"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SAMPLES_JSON = os.path.join(OUT_DIR, "val_eval_samples_popular.json")
OUT_STATS_JSON   = os.path.join(OUT_DIR, "val_eval_samples_stats.json")
FREQ_FILE        = os.path.join(OUT_DIR, "token_freq.json")

N_SAMPLES = 10

SAFE_MIN_COUNT = 38000

# Чи включати fridge у підрахунок частот токенів?
INCLUDE_FRIDGE_IN_COUNTS = True

# Чи перераховувати частоти заново, навіть якщо файл існує
RECOMPUTE_FREQS = False

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

enc = tiktoken.get_encoding("gpt2")

def load_texts_from_json_array(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [obj["text"] for obj in data]

def load_texts_fridge_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [row["instruction"] + " " + row["response"] for row in data]

def token_ids(text: str) -> List[int]:
    return enc.encode(text, disallowed_special=())

def count_token_ids(texts: List[str], desc: str) -> Counter:
    cnt = Counter()
    for t in tqdm(texts, desc=desc):
        cnt.update(token_ids(t))
    return cnt

def save_freq(counter: Counter, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counter, f)

def load_freq(path: str) -> Counter:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Counter({int(k): v for k, v in data.items()})

def get_freq_counter(tiny_train_texts, tiny_val_texts, fridge_texts) -> Counter:
    if not RECOMPUTE_FREQS and os.path.exists(FREQ_FILE):
        print(f"Loading token frequencies from {FREQ_FILE} ...")
        return load_freq(FREQ_FILE)

    print("Counting GPT-2 token frequencies (this may take a while)...")
    freq_counter = Counter()
    freq_counter.update(count_token_ids(tiny_train_texts, "Counting tokens: TinyStories train"))
    freq_counter.update(count_token_ids(tiny_val_texts,   "Counting tokens: TinyStories val"))
    if INCLUDE_FRIDGE_IN_COUNTS and fridge_texts:
        freq_counter.update(count_token_ids(fridge_texts, "Counting tokens: fridge (optional)"))

    save_freq(freq_counter, FREQ_FILE)
    print(f"Token frequencies saved to {FREQ_FILE}")
    return freq_counter

def min_token_freq_in_text(text: str, freq: Dict[int, int]) -> int:
    ids = token_ids(text)
    if not ids:
        return 10**9
    return min(freq.get(i, 0) for i in ids)

def main():
    print("Loading datasets...")
    tiny_train_texts = load_texts_from_json_array(TINYSTORIES_TRAIN_JSON)
    tiny_val_texts   = load_texts_from_json_array(TINYSTORIES_VAL_JSON)

    print(f"TinyStories: train={len(tiny_train_texts)}, val={len(tiny_val_texts)}")

    fridge_texts: List[str] = []
    if INCLUDE_FRIDGE_IN_COUNTS and os.path.exists(FRIDGE_JSON):
        fridge_texts = load_texts_fridge_json(FRIDGE_JSON)
        print(f"Fridge texts: {len(fridge_texts)}")

    # Отримуємо або завантажуємо попередньо збережені частоти токенів
    freq_counter = get_freq_counter(tiny_train_texts, tiny_val_texts, fridge_texts)

    total_unique_tokens = len(freq_counter)
    print(f"Total unique GPT-2 tokens in counts: {total_unique_tokens}")

    print("Scoring validation stories by minimum token frequency...")
    val_with_scores: List[Tuple[int, str, int]] = []
    for idx, text in enumerate(tqdm(tiny_val_texts, desc="Scanning val")):
        m = min_token_freq_in_text(text, freq_counter)
        val_with_scores.append((idx, text, m))

    safe_val = [(idx, text, m) for (idx, text, m) in val_with_scores if m >= SAFE_MIN_COUNT]
    print(f"Validation stories total: {len(tiny_val_texts)}")
    print(f"Stories that pass SAFE_MIN_COUNT={SAFE_MIN_COUNT}: {len(safe_val)}")

    if len(safe_val) < N_SAMPLES:
        print(f"WARNING: Only {len(safe_val)} stories satisfy SAFE_MIN_COUNT={SAFE_MIN_COUNT}, "
              f"but N_SAMPLES={N_SAMPLES}. Consider lowering SAFE_MIN_COUNT or increasing val size.")

    safe_val.sort(key=lambda x: x[2], reverse=True)

    TOP_POOL = min(200, len(safe_val))
    pool = safe_val[:TOP_POOL]
    random.shuffle(pool)
    chosen = pool[:N_SAMPLES]

    chosen_texts = [{"text": t} for (_, t, _) in chosen]
    with open(OUT_SAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(chosen_texts, f, ensure_ascii=False, indent=2)

    stats = {
        "safe_min_count_threshold": SAFE_MIN_COUNT,
        "n_val_total": len(tiny_val_texts),
        "n_safe_candidates": len(safe_val),
        "n_chosen": len(chosen),
        "random_seed": RANDOM_SEED,
        "top_pool_used": TOP_POOL,
        "chosen_min_freqs": [m for (_, _, m) in chosen],
        "chosen_indices_in_val": [idx for (idx, _, _) in chosen],
    }
    with open(OUT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chosen)} evaluation samples to: {OUT_SAMPLES_JSON}")
    print(f"Saved selection stats to: {OUT_STATS_JSON}")
    if chosen:
        print("Sample min-freqs:", stats["chosen_min_freqs"])
        print("Indices in original val:", stats["chosen_indices_in_val"])

if __name__ == "__main__":
    main()
