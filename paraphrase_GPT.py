import fitz  # PyMuPDF
import openai
import openai.error
import json
import re
from time import sleep
import time
from dotenv import load_dotenv
import os
import base64
import matplotlib.pyplot as plt

load_dotenv()

PDF_PATH = os.getenv('PDF_PATH')
MODEL_NAME = os.getenv('MODEL_NAME_V2')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
openai.api_key = os.getenv("OPENAI_API_KEY_2")
OPENAI_KEY_2 = os.getenv("OPENAI_API_KEY_2")
USAGE_FILE = "api_usage.json"
MAX_REQUESTS = 200
ORIG_INPUT_JSON = os.getenv('ORIG_INPUT_JSON')

def encode_file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except openai.error.RateLimitError:
            print(f"⏳ Rate limit hit, sleeping 20s (attempt {attempt+1})")
            sleep(20)
    switch_api_key()
    raise RuntimeError("Rate limit hit too many times.")

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


def pdf_to_page_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        # pix.save(f"page_{page_num+1}.png")
        image_bytes = pix.tobytes("png")
        image_b64 = base64.b64encode(image_bytes).decode()
        
        images.append(image_b64)
        
    return images

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

def switch_api_key(limit=200):
    global request_count
    if request_count >= limit:
        # Перемикання на наступний ключ
        openai.api_key = OPENAI_KEY_2  # Новий ключ
        print("🔑 Switched API key.")
        request_count = 0  # Скидання лічильника для нового ключа
    return openai.api_key

# Зберегти після кожного запиту
def increment_request_count():
    global request_count
    request_count += 1
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": request_count}, f)
    if request_count == MAX_REQUESTS:
        switch_api_key(openai.api_key, limit=MAX_REQUESTS)

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
