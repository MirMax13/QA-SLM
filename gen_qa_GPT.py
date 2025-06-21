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
MODEL_NAME_2 = os.getenv('MODEL_NAME_V2.2')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_KEY_2 = os.getenv("OPENAI_API_KEY_2")
USAGE_FILE = "api_usage.json"
MAX_REQUESTS = 200

def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except openai.error.RateLimitError:
            print(f"⏳ Rate limit hit, sleeping 20s (attempt {attempt+1})")
            sleep(20)
    switch_model()
    # raise RuntimeError("Rate limit hit too many times.")
    try:
        return call_func(*args, **kwargs)
    except Exception as e:
        print(f"❌ GPT call failed after model switch: {e}")
        return None

def call_vision_chat_primary(image_b64: str, prev_text: str = ""):
    content = [
        {"type": "text", "text": (
            "This page contains multiple safety and operational instructions. "
            "Please extract as many detailed, informative, and specific Q&A pairs as possible based strictly on the text. "
            "Use one pair per item or logical group, but be thorough and exhaustive in your responses. "
            "Do not just repeat basic information—elaborate on each answer to ensure completeness. "
            "If necessary, ask more detailed questions to cover multiple aspects. "
            "Output should be in format: Q: ... A: ... with elaborations where applicable."
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    if prev_text:
        content.insert(1, {"type": "text", "text": f"Previous context:\n{prev_text}"})

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant that generates comprehensive, informative, and accurate Q&A pairs from product manuals. "
             "Your answers should not be overly simplistic but should elaborate on the information, providing context and additional details when appropriate."},
            {"role": "user", "content": content}
        ],
        max_tokens=3000
    )
    return response["choices"][0]["message"]["content"]

def call_vision_chat_secondary(image_b64: str, prev_qa_text: str):
    content = [
        {"type": "text", "text": (
            "You previously extracted Q&A pairs from this manual page. Now continue extracting **all remaining** specific Q&A pairs. "
            "Be thorough and exhaustive. Even small details or less prominent points should be converted into QA. "
            "Avoid repeating existing ones. If any information is important but hasn't been included, create new pairs with more depth and context."
        )},
        {"type": "text", "text": f"Previously extracted content:\n{prev_qa_text}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant continuing a QA extraction task. Provide all relevant and detailed Q&A pairs. Ensure each new question adds value and is based on context."},
            {"role": "user", "content": content}
        ],
        max_tokens=2000
    )
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

def switch_model():
    global MODEL_NAME
    MODEL_NAME = MODEL_NAME_2
    print(f"🔄 Switched model to {MODEL_NAME}")

def increment_request_count(): #TODO: probably delete this function
    global request_count
    request_count += 1
    with open(USAGE_FILE, "w") as f:
        json.dump({"count": request_count}, f)

def save_original_qas(new_qas, file_path="original_qas.json"):
    # Зчитуємо існуючі пари, якщо файл вже є
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_qas = json.load(f)
    else:
        existing_qas = []

    # Додаємо нові пари
    existing_qas.extend(new_qas)

    # Перезаписуємо файл
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_qas, f, ensure_ascii=False, indent=2)


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
    blocks = []
    prev_text = ""
    start_time = time.time()
    for idx, image_b64 in enumerate(page_images):
        print(f"🖼️ Processing page {idx+1}/{len(page_images)}")
        print(f"Image {idx+1} size: {len(image_b64) / 1024 / 1024:.2f} MB")
        try:
            raw_primary = safe_gpt_call(call_vision_chat_primary,image_b64, prev_text)
            increment_request_count()
            # print("🧾 Primary GPT output:\n", raw_primary)
            qas_primary = parse_qa_pairs(raw_primary)
            raw_secondary = safe_gpt_call(call_vision_chat_secondary,image_b64, raw_primary)
            increment_request_count()
            # print("🧾 Secondary GPT output:\n", raw_secondary)
            qas_secondary = parse_qa_pairs(raw_secondary)

            combined_qas = qas_primary + qas_secondary
            blocks.append({"qas": combined_qas, "image": image_b64})
            save_original_qas(combined_qas, f"original_qas.json")
            prev_text = raw_primary + "\n\n" + raw_secondary
        except Exception as e:
            print(f"❌ Failed to process page {idx+1}: {e}")

    with open("blocks.json", "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)
    print("Saved blocks.json")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total processing time: {elapsed_time / 60:.2f} minutes")
    with open("token_stats.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
