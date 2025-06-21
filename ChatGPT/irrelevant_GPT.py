
import openai
import json
import re
import time
import os
import matplotlib.pyplot as plt
import random
from config.config import MODEL_NAME, GENERATIVE, USAGE_FILE, OUTPUT_JSON, OUTPUT_JSON_CLEANED, OPENAI_API_KEY
from utils import safe_gpt_call
openai.api_key = OPENAI_API_KEY

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

def load_qas(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_blocks_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and all("context" in block for block in data):
        print(f"📄 Завантажено {len(data)} блоків із {file_path}")
        return [block["context"] for block in data]
    else:
        raise ValueError(f"JSON file {file_path} does not contain valid blocks with 'context' keys")

def parse_qa_pairs(text):
    qas = []
    
    # Спершу класичний формат Q:... A:...
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL)
    blocks = load_blocks_from_json("blocks.json")
    for q, a in qa_blocks: #TODO: maybe delete this loop
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', a.strip().replace("\n", " ")).strip()
        if question and answer:
            if GENERATIVE:
                qas.append({"instruction": question, "response": answer, "tag": "irrelevant"})
            else:
                context = random.choice(blocks)
                qas.append({"context": context, "question": question, "answers": [], "is_impossible": True})

    # Якщо нічого не знайшли — шукаємо формат "1. ... - ..."
    if not qas:
        alt_blocks = re.findall(r"\d+\.\s*(.*?)\n\s*[-•]\s*(.*?)(?=\n\d+\.|\Z)", text, re.DOTALL)
        for q, a in alt_blocks:
            question = q.strip().replace("\n", " ")
            answer = a.strip().replace("\n", " ")
            if question and answer:
                if GENERATIVE:
                    qas.append({"instruction": question, "response": answer, "tag": "irrelevant"})
                else:
                    context = random.choice(blocks)
                    qas.append({"context": context, "question": question, "answers": [], "is_impossible": True})
    print(f"🔍 Found {len(qas)} QA pairs")
    return qas

def generate_irrelevant_qas(n=50, batch_size=10, used_questions=None):
    qas = []
    global request_count
    batches = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    if used_questions is None:
        used_questions = set()
    for b_idx, (start, end) in enumerate(batches):
        count = end - start
        print(f"🔄 Generating irrelevant QAs batch {b_idx+1}/{len(batches)} ({count} pairs)")
        prompt = f""" Write {count} unrelated questions that a refrigerator cannot answer, and assign each an appropriate refusal response.
        The questions should be diverse and cover different topics unrelated to refrigerators.
        
        Use exactly this format (with Q1:, Q2:, etc.):
        Q1: [question]
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]
        
        Q2: [next question]
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]
        
        Do not repeat any of the following questions: {', '.join(used_questions)}"""
        
        messages = [
            {"role": "system", "content": "You are a QA data generator."},
            {"role": "user", "content": prompt}
        ]
        text = safe_gpt_call(call_lm,messages, max_tokens=1024)
        batch_qas = parse_qa_pairs(text)
        for qa in batch_qas:
            if GENERATIVE: #TODO: merge to one
                used_questions.add(qa['instruction'])
            else:
                used_questions.add(qa['question'])

        qas.extend(batch_qas)

        if os.path.exists("irrelevant_qa.json"):
            with open("irrelevant_qa.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.extend(batch_qas)
            qas = existing_data
        with open("irrelevant_qa.json", "w", encoding="utf-8") as f:
            json.dump(qas, f, ensure_ascii=False, indent=2)

    return qas[:n], used_questions

def main():
    global request_count
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            request_count = json.load(f).get("count", 0)
    else:
        request_count = 0

    global token_stats
    token_stats = []
    start_time = time.time()

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(OUTPUT_JSON_CLEANED, "r", encoding="utf-8") as f:
        dataset_cleaned = json.load(f)
    
    print("\n🚫 Generating irrelevant questions...")
    irrelevant_qas,_ = generate_irrelevant_qas(n=150, batch_size=16)
    
    dataset.extend(irrelevant_qas)
    dataset_cleaned.extend(irrelevant_qas)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
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
