# Complete and corrected script with proper use of ChatGPT API for all processing steps

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
openai.api_key = os.getenv("OPENAI_API_KEY")
USAGE_FILE = "api_usage.json"
MAX_REQUESTS = 195


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
    raise RuntimeError("Rate limit hit too many times.")

def call_vision_chat_primary(image_b64: str, prev_text: str = ""):
    content = [
        {"type": "text", "text": (
            "This page contains multiple safety and operational instructions. "
            "Please extract as many meaningful and specific Q&A pairs as possible based strictly on the text. "
            "Use one pair per item or logical group. Output should be in format: Q: ... A: ..."
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    if prev_text:
        content.insert(1, {"type": "text", "text": f"Previous context:\n{prev_text}"})

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant that generates helpful, complete, and accurate Q&A pairs from product manuals. "
             "Do not omit useful content. Provide detailed answers whenever possible. Format: Q: ... A: ..."},
            {"role": "user", "content": content}
        ],
        max_tokens=2000
    )
    return response["choices"][0]["message"]["content"]


def call_vision_chat_secondary(image_b64: str, prev_qa_text: str):
    content = [
        {"type": "text", "text": (
            "You previously extracted Q&A pairs from this manual page. Now continue extracting **all remaining** specific Q&A pairs. "
            "Be thorough and exhaustive. Even small details or less prominent points should be converted into QA. Do not repeat existing ones. Format: Q: ... A: ..."

        )},
        {"type": "text", "text": f"Previously extracted content:\n{prev_qa_text}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant continuing a QA extraction task. Avoid repeating any questions already asked. "
             "Make each new Q&A pair relevant and specific to the manual page."},
            {"role": "user", "content": content}
        ],
        max_tokens=1500
    )
    return response["choices"][0]["message"]["content"]


def call_lm(messages, model=MODEL_NAME, max_tokens=512, temperature=0.3):
    global request_count
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


def parse_qa_pairs(text):
    qas = []
    
    # Спершу класичний формат Q:... A:...
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL)
    for q, a in qa_blocks:
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', a.strip().replace("\n", " ")).strip()
        if question and answer:
            qas.append({"instruction": question, "response": answer, "tag": "good"})

    # Якщо нічого не знайшли — шукаємо формат "1. ... - ..."
    if not qas:
        alt_blocks = re.findall(r"\d+\.\s*(.*?)\n\s*[-•]\s*(.*?)(?=\n\d+\.|\Z)", text, re.DOTALL)
        for q, a in alt_blocks:
            question = q.strip().replace("\n", " ")
            answer = a.strip().replace("\n", " ")
            if question and answer:
                qas.append({"instruction": question, "response": answer, "tag": "good"})

    print(f"🔍 Found {len(qas)} QA pairs")
    return qas


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


def generate_irrelevant_qas(n=50, batch_size=10):
    qas = []
    global request_count
    batches = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    for b_idx, (start, end) in enumerate(batches):
        if request_count >= MAX_REQUESTS:
            print("🚫 Reached request limit. Saving and exiting.")
            break
        count = end - start
        print(f"🔄 Generating irrelevant QAs batch {b_idx+1}/{len(batches)} ({count} pairs)")
        prompt = f""" Write {count} unrelated questions that a refrigerator cannot answer, and assign each an appropriate refusal response.
        The questions should be diverse and cover different topics unrelated to refrigerators.
        
        Use exactly this format (with Q1:, Q2:, etc.):
        Q1: [question]
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]
        
        Q2: [next question]
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]"""
        messages = [
            {"role": "system", "content": "You are a QA data generator."},
            {"role": "user", "content": prompt}
        ]
        text = safe_gpt_call(call_lm,messages, max_tokens=512)
        batch_qas = parse_qa_pairs(text)
        qas.extend(batch_qas)

    return qas[:n]


def filter_qa_candidates(qas, batch_size=35, log_file="filtered_pairs_log.json"):
    if not qas:
        return []

    cleaned = []
    total = len(qas)
    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]

    # Список для логування пар, які були на фільтрі
    filtered_pairs = []

    for b_idx, batch in enumerate(batches):
        print(f"🔍 Filtering batch {b_idx+1}/{len(batches)} with {len(batch)} pairs")
        text = "\n".join([f"{i+1}. Q: {qa['instruction']}\n   A: {qa['response']}" for i, qa in enumerate(batch)])
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

        # Логування пар, які йшли на фільтрацію
        filtered_pairs.extend(batch)

        result = safe_gpt_call(call_lm,messages, max_tokens=1024,temperature=0.7)
        indices = set(int(i.strip()) for i in re.findall(r"\d+", result))
        print("🔍 Filtered indices:", indices)
        cleaned.extend([batch[i - 1] for i in indices if 0 < i <= len(batch)])
        increment_request_count()
        # Виведення кількості пар після фільтрації
        print(f"✅ {len(cleaned)} pairs kept after filtering")

        # Записуємо фільтровані пари в JSON
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(filtered_pairs, f, ensure_ascii=False, indent=2)

    return cleaned


# Зберегти після кожного запиту
def increment_request_count():
    global request_count
    request_count += 1
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": request_count}, f)

def load_data_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    global request_count
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            request_count = json.load(f).get("count", 0)
    else:
        request_count = 0

    global token_stats
    token_stats = []
    page_images = pdf_to_page_images(PDF_PATH)
    # prev_text = ""
    start_time = time.time()
    dataset = load_data_from_json("datasets/open_chat/generative/fridge_dataset_v2.4_small.json")
    dataset_cleaned = []
    for idx, image_b64 in enumerate(page_images):
        print(f"🖼️ Processing page {idx+1}/{len(page_images)}")
        print(f"Image {idx+1} size: {len(image_b64) / 1024 / 1024:.2f} MB")
        try:
            filtered_qas = filter_qa_candidates(dataset)
            dataset_cleaned.extend(filtered_qas)
            print(f"➕ Added {len(dataset)} QA pairs from block {idx+1}")
            print(f"✅ {len(filtered_qas)} kept after filtering")
            
        except Exception as e:
            print(f"❌ Failed to process page {idx+1}: {e}")

    # with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    #     json.dump(dataset, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
        json.dump(dataset_cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(dataset)} total entries (full)")
    print(f"✅ Saved {len(dataset_cleaned)} cleaned entries (after filtering)")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total processing time: {elapsed_time / 60:.2f} minutes")
    with open("token_stats.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
