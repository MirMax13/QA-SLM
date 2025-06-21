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
from datetime import datetime

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
            "This page contains multiple safety and operational instructions.\n"
            "Please first extract the full text content of this manual page (as if OCR).\n" #Was "also If there is a table, convert each row into a full sentence when building the text content."
            "Then generate as many detailed, helpful, and **fully informative** question–answer pairs as possible.\n"
            "Each **answer must be copied exactly** from the text (no paraphrasing)."
            "\n\n"
            "⚠️ When selecting the answer span:\n"
            "- Prefer the **entire bullet point**, sentence, or numbered item that contains the answer.\n"
            "- If an item contains multiple clauses, include all if they are related.\n"
            "- Do NOT shorten useful text — preserve context.\n"
            "- - If the item is too short (e.g. a single sentence or phrase), include the full relevant bullet, point, or numbered section it belongs to.\n\n"
            "Return JSON like:\n"
            "{\n  \"context\": \"...full OCR text...\",\n  \"qas\": [ {\"question\": \"...\", \"answer\": \"...\"}, ... ]\n}"
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    if prev_text:
        content.insert(1, {"type": "text", "text": f"Previous context:\n{prev_text}"})

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant that extracts exact answers from the provided manual context. "
             "The answer should be an exact part of the context with the position marked in the answer."},
            {"role": "user", "content": content}
        ],
        max_tokens=3000
    )
    response_text = response["choices"][0]["message"]["content"].strip()
    if not response_text:
        print("⚠️ Empty GPT response in call_vision_chat_primary.")
        return "", []
    try:
        parsed = json.loads(response_text)
        context = parsed["context"]
        qas_raw = parsed["qas"]
        return context, qas_raw
    except Exception as e:
        print("❌ Failed to parse GPT JSON response:", e)
        return "", []

def call_vision_chat_secondary(image_b64: str, prev_qa_text: str, context: str):
    content = [
        {"type": "text", "text": (
            "You previously extracted Q&A pairs from this manual page. Now continue extracting **all remaining** specific Q&A pairs.\n"
            "Be thorough and exhaustive. Even small details or less prominent points should be converted into QA.\n\n"
            "⚠️ When selecting the answer span:\n"
            "- **Always copy the full sentence or bullet/numbered item** from the context.\n"
            "- If a sentence contains several useful parts, include them all.\n"
            "- Only output the raw JSON. Do not include any commentary, markdown formatting, or explanation.\n"
            "- Do NOT paraphrase or truncate. Copy the full logical statement.\n"
            "- If the answer is too short to be useful, extend it to include the full meaningful sentence or section from the context."
            "Return in JSON:\n\n"
            "{\n  \"qas\": [ {\"question\": \"...\", \"answer\": \"...\"}, ... ]\n}"
        )},
        {"type": "text", "text": f"Previously extracted questions:\n{prev_qa_text}"},
        {"type": "text", "text": f"Context:\n{context}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "You are a documentation assistant continuing a QA extraction task.Your goal is to extract additional high-quality QA pairs that were not in the previous response."},
            {"role": "user", "content": content}
        ],
        max_tokens=2000
    )
    response_text = response["choices"][0]["message"]["content"].strip()
    if not response_text:
        print("⚠️ Empty GPT response in call_vision_chat_secondary.")
        return []
    try:
        parsed = json.loads(response["choices"][0]["message"]["content"])
        return parsed["qas"]
    except Exception as e:
        print("❌ Failed to parse secondary QA JSON:", e)
        with open("error_secondary_response.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Error at {datetime.now().isoformat()} ===\n")
            f.write(response_text)
        print("📩 Raw response saved to error_secondary_response.txt")
        return []

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

def find_answer_start_fuzzy(context, answer):
    # Escape special regex characters
    idx = context.find(answer)
    if idx != -1:
        return idx
    
    # Пробуємо з очищенням пробілів і лапок
    answer_clean = re.sub(r'\s+', ' ', answer).strip().strip('"\'')
    context_clean = re.sub(r'\s+', ' ', context)

    idx_clean = context_clean.find(answer_clean)
    if idx_clean != -1:
        print("🧪 Found approximate match for:", answer_clean)
        return idx_clean

    return -1

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
    n = 5
    for idx, image_b64 in enumerate(page_images[n:]):
        print(f"🖼️ Processing page {idx+1+n}/{len(page_images)}")
        print(f"Image {idx+1+n} size: {len(image_b64) / 1024 / 1024:.2f} MB")
        try:
            context, qas_raw = safe_gpt_call(call_vision_chat_primary, image_b64, prev_text)
            increment_request_count()
            qas_primary = []
            print(f" Found {len(qas_raw)} primary Q&A pairs")

            for qa in qas_raw:
                answer = qa["answer"].strip()
                answer_start = find_answer_start_fuzzy(context, answer)
                if answer_start == -1:
                    print(f"⚠️ Answer not found in context: {answer}")
                    # continue
                qas_primary.append({
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": [{"text": answer, "answer_start": answer_start}],
                    "is_impossible": False
                })
            
            prev_questions_text = "\n".join([f"Q: {qa['question'].strip()}" for qa in qas_raw])
            qas_secondary_raw = safe_gpt_call(call_vision_chat_secondary, image_b64, prev_questions_text, context)
            increment_request_count()
            qas_secondary = []

            print(f" Found {len(qas_secondary_raw)} secondary Q&A pairs")
            for qa in qas_secondary_raw:
                answer = qa["answer"].strip()
                answer_start = find_answer_start_fuzzy(context, answer)
                if answer_start == -1:
                    print(f"⚠️ Answer not found in context (secondary): {answer}")
                    # continue
                qas_secondary.append({
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": [{"text": answer, "answer_start": answer_start}],
                    "is_impossible": False
                })

            # print("🧾 Secondary GPT output:\n", raw_secondary)

            combined_qas = qas_primary + qas_secondary
            blocks.append({"context": context, "qas": combined_qas, "image": image_b64})
            save_original_qas(combined_qas, f"original_qas.json")
            prev_questions = [qa["question"].strip() for qa in qas_raw + qas_secondary_raw]
            prev_text = "\n".join([f"Q: {q}" for q in prev_questions])
        except Exception as e:
            print(f"❌ Failed to process page {idx+1+n}: {e}")

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
