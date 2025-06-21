
import openai
import json
import time
import os
import matplotlib.pyplot as plt
from config.config import GENERATIVE, OUTPUT_JSON, OUTPUT_JSON_CLEANED, OPENAI_API_KEY
from utils import safe_gpt_call, call_lm,parse_qa_pairs
openai.api_key = OPENAI_API_KEY

def load_blocks_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and all("context" in block for block in data):
        print(f"📄 Завантажено {len(data)} блоків із {file_path}")
        return [block["context"] for block in data]
    else:
        raise ValueError(f"JSON file {file_path} does not contain valid blocks with 'context' keys")

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
        batch_qas = parse_qa_pairs(text, status="irrelevant")
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
    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
