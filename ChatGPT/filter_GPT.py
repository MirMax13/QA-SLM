
import openai
import openai.error
import json
import re
from time import sleep
import time
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import random

load_dotenv()

PDF_PATH = os.getenv('PDF_PATH')
MODEL_NAME = os.getenv('MODEL_NAME_V2')
MODEL_NAME_2 = os.getenv('MODEL_NAME_V2.2')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
openai.api_key = os.getenv("OPENAI_API_KEY")
USAGE_FILE = "api_usage.json"
MAX_REQUESTS = 200
GENERATIVE = True

def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except openai.error.RateLimitError:
            print(f"⏳ Rate limit hit, sleeping 20s (attempt {attempt+1})")
            sleep(20)
    switch_model()  # Switch model if rate limit is hit
    # raise RuntimeError("Rate limit hit too many times.")
    try:
        return call_func(*args, **kwargs)
    except Exception as e:
        print(f"❌ GPT call failed after model switch: {e}")
        return None

def call_lm(messages, model=MODEL_NAME, max_tokens=512, temperature=0.7):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    print(f"📊 Tokens used: prompt={prompt_tokens}, completion={completion_tokens}")

    # Save stats to global list
    token_stats.append({
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "messages": messages[-1]["content"][:100]  # just preview
    })

    return response["choices"][0]["message"]["content"]

def filter_qa_candidates(qas, batch_size=35):
    if not qas:
        return []

    cleaned = []
    total = len(qas)
    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]

    n = 0
    for b_idx, batch in enumerate(batches[n:]): # Пропускаємо перші n партій
        print(f"🔍 Filtering batch {b_idx+1+n}/{len(batches)} with {len(batch)} pairs")
        if GENERATIVE:
            text = "\n".join([f"{i+1}. Q: {qa['instruction']}\n   A: {qa['response']}" for i, qa in enumerate(batch)])
        else:
            text = "\n".join([f"{i+1}. Q: {qa['question']}\n   A: {qa['answers'][0]['text']}" for i, qa in enumerate(batch)])
        prompt = f"""
        You are reviewing QA pairs from a refrigerator manual. Accept all that are:
        - understandable and relevant (minor typos are OK)
        - logical and informative, even if not perfectly structured
        - not exact duplicates, but paraphrases or minor rewordings are acceptable if they offer new information, additional context, or improved clarity

        Reject only if the pair is irrelevant, confusing, or lacking value. Return a list of valid indices like: 1, 2, 3, 4, 5.
        {text}
        """
        messages = [
            {"role": "system", "content": "You are a QA data cleaner."},
            {"role": "user", "content": prompt}
        ]

        result = safe_gpt_call(call_lm,messages, max_tokens=1024)
        indices = set(int(i.strip()) for i in re.findall(r"\d+", result))
        print("🔍 Filtered indices:", indices)
        new_cleaned = [batch[i - 1] for i in indices if 0 < i <= len(batch)]
        increment_request_count()
        print(f"✅ {len(new_cleaned)} pairs kept after filtering")

        # Зберігаємо дані в JSON файл
        if os.path.exists(OUTPUT_JSON_CLEANED):
            with open(OUTPUT_JSON_CLEANED, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            cleaned = existing_data.copy()  # Копіюємо існуючі дані
            existing_data.extend(new_cleaned)  # Додаємо нові елементи до вже існуючого списку
            with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        else:
            with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
                json.dump(new_cleaned, f, ensure_ascii=False, indent=2)

        cleaned.extend(new_cleaned)
        print("Cleaned length:", len(cleaned))

    print(f"✅ Total kept after filtering: {len(cleaned)} out of {total} pairs")
    return cleaned

def switch_model():
    global MODEL_NAME
    MODEL_NAME = MODEL_NAME_2
    print(f"🔄 Switched model to {MODEL_NAME}")

def increment_request_count(): #TODO: probably delete this function
    global request_count
    request_count += 1
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": request_count}, f)

def main():
    random.seed(42)
    global request_count
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            request_count = json.load(f).get("count", 0)
    else:
        request_count = 0

    global token_stats
    token_stats = []
    start_time = time.time()

    dataset = []
    dataset_cleaned = []
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"🔄 Loaded {len(dataset)} existing entries from {OUTPUT_JSON}")

    random.shuffle(dataset)  # Перемішуємо для різноманітності
    filtered_qas = filter_qa_candidates(dataset, batch_size=35)
    # dataset_cleaned = filtered_qas

    # if os.path.exists(OUTPUT_JSON_CLEANED):
    #     with open(OUTPUT_JSON_CLEANED, "r", encoding="utf-8") as f:
    #         existing_cleaned = json.load(f)
    #     dataset_cleaned.extend(existing_cleaned)
    #     print(f"🔄 Loaded {len(existing_cleaned)} existing cleaned entries from {OUTPUT_JSON_CLEANED}")

    # print(f"\n✅ Saved {len(dataset)} total entries (full)")
    # print(f"✅ Saved {len(dataset_cleaned)} cleaned entries (after filtering)")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total processing time: {elapsed_time / 60:.2f} minutes")
    if os.path.exists("token_stats.json"):
        with open("token_stats.json", "r", encoding="utf-8") as f:
            existing_stats = json.load(f)
        token_stats.extend(existing_stats)
        print(f"🔄 Loaded {len(existing_stats)} existing token stats from token_stats.json")
    with open("token_stats.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
