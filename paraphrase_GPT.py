import openai
import openai.error
import json
import re
from time import sleep
import time
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()

PDF_PATH = os.getenv('PDF_PATH')
MODEL_NAME = os.getenv('MODEL_NAME_V2')
MODEL_NAME_2 = os.getenv('MODEL_NAME_V2.2')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
openai.api_key = os.getenv("OPENAI_API_KEY_2")
OPENAI_KEY_2 = os.getenv("OPENAI_API_KEY_2")
USAGE_FILE = "api_usage.json"
MAX_REQUESTS = 200
ORIG_INPUT_JSON = os.getenv('ORIG_INPUT_JSON')

def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except openai.error.RateLimitError:
            print(f"⏳ Rate limit hit, sleeping 20s (attempt {attempt+1})")
            sleep(20)
    switch_model()  # Switch model if rate limit is hit
    # raise RuntimeError("Rate limit hit too many times.")

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

def generate_paraphrases(text, is_question=True, n=3):
    role = "question" if is_question else "answer"
    prompt = f"Generate {n} diverse paraphrases of the following {role}, preserving its meaning:\n\n\"{text}\"\n\n"
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates diverse, high-quality paraphrases of questions and answers from manuals. Do not invent information, stay on topic, and vary structure and wording."},
        {"role": "user", "content": prompt}
    ]
    raw = safe_gpt_call(call_lm,messages)
    lines = [re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', l.strip("-• ")) for l in raw.strip().splitlines() if l.strip()]
    print(f"🔄 Generated {len(lines)} paraphrases for {role}")
    return lines[:n]

def switch_model():
    global MODEL_NAME
    MODEL_NAME = MODEL_NAME_2
    print(f"🔄 Switched model to {MODEL_NAME}")

# Зберегти після кожного запиту
def increment_request_count(): #TODO: probably delete this function
    global request_count
    request_count += 1
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": request_count}, f)

def load_qas(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def main():
    global request_count
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            request_count = json.load(f).get("count", 0)
    else:
        request_count = 0

    global token_stats
    token_stats = []
    blocks = load_qas(ORIG_INPUT_JSON)
    start_time = time.time()

    dataset = []
    n = 0
    for idx, block in enumerate(blocks[n:]): # Пропускаємо перші n блоків
        print(f"\n🔷 Processing block {idx+1+n}/{len(blocks)}")
        all_qas = []

        if isinstance(block, dict) and 'instruction' in block and 'response' in block:
            paraphrased_qs = generate_paraphrases(block['instruction'], is_question=True, n=5)
            increment_request_count()

            paraphrased_as = generate_paraphrases(block['response'], is_question=False, n=3)
            increment_request_count()

            # Додаємо оригінальні пари та перефразовані варіанти
            for pq in [block['instruction']] + paraphrased_qs:
                for pa in [block['response']] + paraphrased_as:
                    all_qas.append({"instruction": pq, "response": pa, "tag": "good"})

        dataset.extend(all_qas)

        # Зберігаємо дані в JSON файл
        if os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.extend(all_qas)
            dataset = existing_data
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"🔷 Processed block {idx+1+n}/{len(blocks)}: {len(all_qas)} QA pairs")
        
        print(f"➕ Added {len(all_qas)} QA pairs from block {idx+1+n}")


    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


    print(f"\n✅ Saved {len(dataset)} total entries (full)")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total processing time: {elapsed_time / 60:.2f} minutes")
    with open("token_stats.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
