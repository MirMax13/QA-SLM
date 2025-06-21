import openai
import json
import re
import time
import os
import matplotlib.pyplot as plt
from config.config import MODEL_NAME, GENERATIVE, ORIG_INPUT_JSON, USAGE_FILE,OUTPUT_JSON,OPENAI_API_KEY
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
        question = 'question'
        answer = 'answers'
        if GENERATIVE:
            question = 'instruction'
            answer = 'response'

        if isinstance(block, dict) and question in block and answer in block:
            paraphrased_qs = generate_paraphrases(block[question], is_question=True, n=6)

            if GENERATIVE:
                paraphrased_as = generate_paraphrases(block[answer], is_question=False, n=3)
                for pq in [block[question]] + paraphrased_qs:
                    for pa in [block[answer]] + paraphrased_as:
                        all_qas.append({"instruction": pq, "response": pa, "tag": "good"})
            else:
                # Додаємо оригінальні пари та перефразовані варіанти
                for pq in [block[question]] + paraphrased_qs:
                    for pa in block[answer]:
                        all_qas.append({
                            "context": block['context'],
                            "question": pq,
                            "answers": [pa],
                            "is_impossible": False
                            })

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

    print(f"\n✅ Saved {len(dataset)} total entries (full)")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total processing time: {elapsed_time / 60:.2f} minutes")
    with open("token_stats.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    plt.figure(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
