import requests
import json
import re
from time import sleep
from dotenv import load_dotenv
import os
import random
from fuzzywuzzy import fuzz

from transformers import AutoTokenizer
load_dotenv()

LM_API_URL = os.getenv('LM_API_URL')
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = os.getenv('MODEL_NAME')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
INSTRUCTION_PATH = os.getenv('INSTRUCTION_PATH')

# ========== AGENT HELPERS ==========
# 2*1*1*300 + 2*5*5*300 = 15600
# 3*1*1*300 + 3*5*3*512 = 23940

def num_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.6-8b-20240522", trust_remote_code=True)
    return len(tokenizer.encode(text))
def call_lm(messages, temperature=0.7, max_tokens=512):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(LM_API_URL, headers=HEADERS, json=payload, timeout=40)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Error during LM call: {e}")
        return ""

# ========== STEP 1: Extract blocks ==========
def load_blocks_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.split(r'🔹 Блок \d+ \(\d+ слів\):\n[-]+\n(.*?)\n[-]+\n', content, flags=re.DOTALL)
    # re.split() повертає масив [префікс, блок1, блок2, ...], тому беремо лише блоки
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    print(f"📄 Завантажено {len(blocks)} блоків із {file_path}")
    return blocks

# ========== STEP 2: QA Generation ==========
def generate_qa_pairs(block_text):
    prompt = f"""
    <|im_start|>system
    You are a QA dataset generator. You are generating training data for a SQuAD-style extractive QA model.

    Your task is to create question–answer pairs from the provided block of instructional text.

    Instructions:
    - The **answer** must be an exact span from the text (no paraphrasing). Copy it word for word as it appears.
    - The **question** must be natural, human-like, and clearly related to the answer. Do NOT copy text from the passage into the question.
    - Always prefer complete and informative answers — select full sentences or list items when possible.
    - If a sentence contains a URL or parentheses, include them in the answer if they are part of the same sentence.
    - Avoid generic answers like "Yes" or "No".
    - Generate 3–5 good question–answer pairs per block if possible.

    Output format:
    Q: [question]
    A: [exact span from text]
    Q: ...
    A: ...

    The instructional text:
    \"\"\"{block_text}\"\"\"
    <|im_end|>
    <|im_start|>user
    Generate QA pairs<|im_end|>
    <|im_start|>assistant
    """

    if len(prompt) > 12000:
        print("⚠️ Prompt too long")
    content = call_lm([{"role": "user", "content": prompt}])
    qas = parse_qa_pairs(content)

    # Логування питань та відповідей
    for qa in qas:
        # print(f"Блок: {block_text}")
        print(f"Запитання: {qa['question']}")
        print(f"Відповідь: {qa['answer']}")

    return qas

# ========== STEP 3: Parse QA ==========
def parse_qa_pairs(text):
    qas = []
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL) #TODO: check if this regex works
    for q, a in qa_blocks:
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = a.strip()
        if question and answer:
            qas.append({
                "question": question,
                "answer": answer,
            })
    print(f"🔍 Found {len(qas)} QA pairs")
    return qas

# ========== STEP 4: Paraphrasing ==========
def generate_paraphrases(text, is_question=True, n=3):
    role = "question" if is_question else "answer"
    prompt = f"""
    <|im_start|>system
    You are a helpful assistant.
    Generate {n} diverse paraphrases of the following {role}, preserving its meaning.
    Feel free to rephrase it in various ways while preserving its intent.
    {role.capitalize()}: "{text}"
    <|im_end|>
    <|im_start|>user
    Paraphrase<|im_end|>
    <|im_start|>assistant
    """
    if len(prompt) > 12000:
        print("⚠️ Prompt too long")
    raw = call_lm([{"role": "user", "content": prompt}], max_tokens=512)
    lines = [re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', l.strip("-• ")) for l in raw.strip().splitlines() if l.strip()]
    print(f"🔄 Generated {len(lines)} paraphrases for {role}")
    return lines[:n]

# ========== STEP 5: Irrelevant QA ==========

def generate_irrelevant_qas(n=50, batch_size=10):
    qas = []
    batches = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    
    for b_idx, (start, end) in enumerate(batches):
        count = end - start
        print(f"🔄 Generating irrelevant QAs batch {b_idx+1}/{len(batches)} ({count} pairs)")
        
        prompt = f"""
        <|im_start|>system
        You are a QA data generator. 
        Write {count} unrelated questions that a refrigerator cannot answer, and assign each an appropriate refusal response.
        The questions should be diverse and cover different topics unrelated to refrigerators.
        
        Use exactly this format (with Q1:, Q2:, etc.):
        Q1: [question]
        A:I cannot help with [reason]
        
        Q2: [next question]
        A: I cannot help with [reason]
        <|im_end|>
        <|im_start|>user
        Generate unrelated QA<|im_end|>
        <|im_start|>assistant
        """
        
        if num_tokens(prompt) > 8000:
            print(f"⚠️ Generation prompt too long in batch {b_idx+1}")
            
        text = call_lm([{"role": "user", "content": prompt}], max_tokens=512)
        batch_qas = parse_qa_pairs(text)
        qas.extend(batch_qas)
        
        print(f"✅ Generated {len(batch_qas)} pairs in batch {b_idx+1}")
        sleep(1)  # Add small delay between batches
    
    return qas[:n]  # Ensure we return exactly n pairs


def filter_qa_candidates(qas, batch_size=35):
    if not qas:
        return []
    
    cleaned = []
    total = len(qas)
    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]
    
    for b_idx, batch in enumerate(batches):
        print(f"🔍 Filtering batch {b_idx+1}/{len(batches)} with {len(batch)} pairs")
        print(batch[0])
        text = "\n".join([f"{i+1}. Q: {qa['question']}\n   A: {qa['answers'][0]['text']}" for i, qa in enumerate(batch)])
        prompt = f"""
        <|im_start|>system
        You are a QA data cleaner. Review the following question-answer pairs and keep all that are:
        - grammatically correct
        - meaningful and relevant to a refrigerator manual
        - not exact duplicates (slightly reworded paraphrases are OK)

        Return a list of indices (e.g., 1, 2, 3, 5, 6).
        <|im_end|>
        <|im_start|>user
        {text}
        <|im_end|>
        <|im_start|>assistant
        """
        if num_tokens(prompt) > 8000:
            print(f"⚠️ Filtering prompt too long in batch {b_idx+1}")
        result = call_lm([{"role": "user", "content": prompt}])
        indices = set(int(i.strip()) for i in re.findall(r"\d+", result))
        cleaned.extend([batch[i - 1] for i in indices if 0 < i <= len(batch)])
    
    return cleaned

def find_fuzzy_span(block, answer, threshold=90):
    best_score = 0
    best_start = -1
    for i in range(len(block) - len(answer)):
        span = block[i:i+len(answer)+20]  # +20 дає трохи більше простору
        score = fuzz.partial_ratio(answer.lower(), span.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_start = i
    return best_start if best_score >= threshold else None

# ========== STEP 6: Main loop ==========
def main():
    # blocks = extract_text_blocks(PDF_PATH)
    blocks = load_blocks_from_txt(INSTRUCTION_PATH)
    dataset = []
    dataset_cleaned = []

    for idx, block in enumerate(blocks):
        print(f"\n🔷 Processing block {idx+1}/{len(blocks)}")
        base_qas = generate_qa_pairs(block)
        all_qas = []

        for qa in base_qas:
            start = find_fuzzy_span(block, qa['answer'])
            if start is not None:
                answer_start = start
            else:
                print(f"⚠️ Відповідь не знайдена в блоці для питання: {qa['question']}")
                continue

            answer_start = block.find(qa['answer'])
            paraphrased_qs = [qa['question']] + generate_paraphrases(qa['question'], is_question=True, n=5)
            sleep(1)
            for pq in paraphrased_qs:
                sample = {
                    "context": block,
                    "question": pq,
                    "answers": [{"text": qa['answer'], "answer_start": answer_start}],
                    "is_impossible": False
                }
                dataset.append(sample)
                all_qas.append(sample)

        # Очищаємо і додаємо до cleaned
        filtered_qas = filter_qa_candidates(all_qas)
        dataset_cleaned.extend(filtered_qas)
        print(f"➕ Added {len(all_qas)} QA pairs from block {idx+1}")
        print(f"✅ {len(filtered_qas)} kept after filtering")
        sleep(2)
    
    # Add irrelevant QAs
    print("\n🚫 Generating irrelevant questions...")
    irrelevant_qas = generate_irrelevant_qas(n=100, batch_size=20)
    for irr in irrelevant_qas:
        random_block = random.choice(blocks)
        dataset.append({
            "context": random_block,
            "question": irr['question'],
            "answers": [],
            "is_impossible": True
        })
        dataset_cleaned.append({
            "context": random_block,
            "question": irr['question'],
            "answers": [],
            "is_impossible": True
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
        json.dump(dataset_cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(dataset)} total entries (full)")
    print(f"✅ Saved {len(dataset_cleaned)} cleaned entries (after filtering)")
if __name__ == "__main__":
    main()
