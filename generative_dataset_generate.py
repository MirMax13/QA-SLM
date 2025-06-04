import fitz  # PyMuPDF
import requests
import json
import re
from time import sleep
from dotenv import load_dotenv
import os

from transformers import AutoTokenizer
load_dotenv()

PDF_PATH = os.getenv('PDF_PATH')
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
def call_lm(messages, temperature=0.7, max_tokens=300):
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
    
# ========== STEP 0: Create blocks ==========
def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    raw_blocks = re.split(r'\n(?=[A-Z][^\n]{0,80}\n)', text)
    blocks = [b.strip() for b in raw_blocks if len(b.strip()) > 100]

    print(f"\n🔍 Всього блоків: {len(blocks)}")

    # Запис у файл
    with open(INSTRUCTION_NAME, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks):
            f.write(f"🔹 Блок {i+1} ({len(block.split())} слів):\n")
            f.write("-" * 60 + "\n")
            f.write(block + "\n")
            f.write("-" * 60 + "\n\n")

    return blocks

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
    You are an intelligent refrigerator that answers user questions related to the provided instruction text.
    Generate 2–3 natural and informative question–answer pairs based on the following manual section.
    If there is an opportunity to ask more questions and answers, this is encouraged (up to 10 pairs)
    Make the answers as complete, helpful, and context-aware as possible.
    Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.
    Ensure each question is distinct and relevant to the text.
    Format:
    Q: ...
    A: ...
    Q: ...
    A: ...
    The text of the instructions:
    \"\"\"{block_text}\"\"\"
    <|im_end|>
    <|im_start|>user
    Generate QA pairs<|im_end|>
    <|im_start|>assistant
    """
    if len(prompt) > 12000:
        print("⚠️ Prompt too long")
    content = call_lm([{"role": "user", "content": prompt}])
    return parse_qa_pairs(content)

# ========== STEP 3: Parse QA ==========
def parse_qa_pairs(text):
    qas = []
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL) #TODO: check if this regex works
    for q, a in qa_blocks:
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', a.strip().replace("\n", " ")).strip()
        if question and answer:
            qas.append({
                "instruction": question,
                "response": answer,
                "tag": "good",
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
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]
        
        Q2: [next question]
        A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason]
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
        sleep(2)  # Add small delay between batches
    
    return qas[:n]  # Ensure we return exactly n pairs


def filter_qa_candidates(qas, batch_size=35):
    if not qas:
        return []
    
    cleaned = []
    total = len(qas)
    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]
    
    for b_idx, batch in enumerate(batches):
        print(f"🔍 Filtering batch {b_idx+1}/{len(batches)} with {len(batch)} pairs")
        text = "\n".join([f"{i+1}. Q: {qa['instruction']}\n   A: {qa['response']}" for i, qa in enumerate(batch)])
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


# ========== STEP 6: Main loop ==========
def main():
    # blocks = extract_text_blocks(PDF_PATH)
    blocks = load_blocks_from_txt(INSTRUCTION_NAME)
    dataset = []
    dataset_cleaned = []

    for idx, block in enumerate(blocks):
        print(f"\n🔷 Processing block {idx+1}/{len(blocks)}")
        base_qas = generate_qa_pairs(block)
        all_qas = []

        for qa in base_qas:
            paraphrased_qs = generate_paraphrases(qa['instruction'], is_question=True, n=5)
            paraphrased_as = generate_paraphrases(qa['response'], is_question=False, n=3)

            for pq in [qa['instruction']] + paraphrased_qs:
                for pa in [qa['response']] + paraphrased_as:
                    all_qas.append({"instruction": pq, "response": pa, "tag": "good"})

        dataset.extend(all_qas)

        # Очищаємо і додаємо до cleaned
        filtered_qas = filter_qa_candidates(all_qas)
        dataset_cleaned.extend(filtered_qas)
        print(f"➕ Added {len(all_qas)} QA pairs from block {idx+1}")
        print(f"✅ {len(filtered_qas)} kept after filtering")
        sleep(2)
    
    # Add irrelevant QAs
    print("\n🚫 Generating irrelevant questions...")
    irrelevant_qas = generate_irrelevant_qas(n=100, batch_size=10)
    for qa in irrelevant_qas:
        qa["tag"] = "irrelevant"
    dataset.extend(irrelevant_qas)
    dataset_cleaned.extend(irrelevant_qas)


    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
        json.dump(dataset_cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(dataset)} total entries (full)")
    print(f"✅ Saved {len(dataset_cleaned)} cleaned entries (after filtering)")
if __name__ == "__main__":
    main()
