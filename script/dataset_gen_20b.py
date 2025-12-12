import os
import json
import re
import time
import torch
from transformers import pipeline

# ================= НАЛАШТУВАННЯ =================
INPUT_TXT = "Instruction_v1.4.txt"
MODEL_ID = "openai/gpt-oss-20b"
OUTPUT_DIR = "dataset_output"

# Параметри
PARAPHRASE_Q_COUNT = 5  # 5 варіантів питання
PARAPHRASE_A_COUNT = 3  # 3 варіанти відповіді
STYLES = ["standard", "boolq", "piqa", "hellaswag"]
FILTER_BATCH_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= МОДЕЛЬ =================
print("🚀 Initializing model...")
try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        model_kwargs={"trust_remote_code": True}
    )
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ================= ФУНКЦІЇ =================

def load_blocks_from_txt(file_path):
    """Ваша функція для парсингу блоків."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Розбиваємо по заголовку блоку
    raw_blocks = re.split(r'🔹 Блок \d+ \(\d+ слів\):\n[-]+\n(.*?)\n[-]+\n', content, flags=re.DOTALL)
    
    # Фільтруємо пусті результати спліту
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    print(f"📄 Завантажено {len(blocks)} блоків із {file_path}")
    return blocks

def save_jsonl(entries, filename):
    """
    Зберігає список об'єктів у формат JSONL (один рядок = один JSON).
    Додає (append) до файлу, якщо він існує.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not isinstance(entries, list):
        entries = [entries]
        
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def llm_call(prompt, max_new=2048, temp=0.7):
    # Формат ChatML
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    try:
        outputs = pipe(
            full_prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=0.95,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        print(f"⚠️ Generation Error: {e}")
        return ""

def parse_json_robust(text):
    """Намагається витягнути JSON зі списку."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return []

# ================= ПРОМПТИ =================

def get_generation_prompt(style, text, existing_qs=""):
    avoid_instr = ""
    if existing_qs:
        avoid_instr = f"Do NOT generate these questions again: {existing_qs}. Find NEW details."

    if style == "standard":
        return f"""
You are an intelligent refrigerator that answers user questions related to the provided instruction text.
Generate as many Question-Answer pairs as possible based on the following manual section.
If there is an opportunity to ask more questions and answers, this is encouraged (up to 5 pairs)
Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.
Ensure each question is distinct and relevant to the text.
{avoid_instr}
Text: \"\"\"{text}\"\"\"
Output ONLY a JSON list: [{{"instruction": "...", "response": "..."}}, ...]
"""
    elif style == "boolq":
        return f"""
You are an intelligent refrigerator that gives helpful, safety-conscious, and detailed answers.
Generate as many as possible  natural question–answer pairs inspired by the BoolQ format.
The user asks if something can or should be done, and you answer "Yes" or "No" with reasoning.
Answers must be realistic, safety-aware, and may include alternatives or extra advice.
Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.
{avoid_instr}
Text: \"\"\"{text}\"\"\"
Output ONLY a JSON list: [{{"instruction": "Can I...?", "response": "No, because..."}}, ...]
"""
    elif style == "piqa":
        return f"""
You are an intelligent refrigerator that helps users make the best practical decisions in everyday home situations.
Generate as many as possible natural PIQA-style question–answer pairs related to the text below.

Each question must:
- describe a small realistic situation involving refrigerators or appliance care
- **always include 2–3 possible actions or options** that a person might choose (some may be wrong)
- never use markers like A/B/C or numbers
- be phrased conversationally and concisely (1 sentence)

Each answer must:
- clearly state which option is correct (or that neither is good)
- explain *why* that choice is best, with short reasoning
- be friendly, helpful, and realistic

Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.

{avoid_instr}
Text: \"\"\"{text}\"\"\"
Output ONLY a JSON list: [{{"instruction": "Is it better to X or Y?", "response": "It is better to X because..."}}, ...]
"""
    elif style == "hellaswag":
        return f"""
You are an intelligent refrigerator that explains the results of user actions.
Generate as many as possible natural question–answer pairs inspired by the HellaSwag format.
Each question should describe an action or event (“What happens if...”, “What will occur when...”), and the answer should describe the likely outcome and reason.
Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.
{avoid_instr}
Text: \"\"\"{text}\"\"\"
Output ONLY a JSON list: [{{"instruction": "What happens if I...?", "response": "The fridge will..."}}, ...]
"""

def get_paraphrase_prompt(text, is_question, count, style="standard"):
    role = "question" if is_question else "answer"
    
    # Додаємо style_hint відповідно до вашого оригінального коду
    style_hint = ""
    if style == "boolq":
        style_hint = "Preserve the 'Yes/No with reasoning' style and tone."
    elif style == "piqa":
        style_hint = "Keep the comparative structure, mentioning multiple options naturally."
    elif style == "hellaswag":
        style_hint = "Preserve the 'what happens if' reasoning tone."

    return f"""
Paraphrase the following {role} in {count} diverse ways. Preserve the meaning exactly.
{style_hint}
Original: "{text}"
Output ONLY a numbered list:
1. ...
2. ...
"""

def get_irrelevant_prompt(batch_size, style):
    style_instructions = ""
    
    if style == "boolq":
            style_instructions = """
Generate yes/no-style questions that are *not* related to refrigerators, food, or home appliances.
Each question should sound natural, like a curiosity a user might have.

Each answer must:
- strictly follow this refusal format:
  A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason].

Examples:
Q: Can humans live on Mars?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about space exploration.

Q: Is it okay to leave a candle burning overnight?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about fire safety.
"""

    elif style == "piqa":
        style_instructions = """
Generate questions comparing 2–3 options, none of which are about refrigerators or food.
Each question should sound natural (no A/B/C labeling).

Each answer must:
- strictly follow this refusal format:
A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason].

Examples:
Q: Should I water plants in the morning or at night?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about gardening.

Q: Is it better to paint walls with a brush or a roller?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about home renovation.
"""

    elif style == "hellaswag":
        style_instructions = """
Generate questions that ask what happens after or as a result of an event, none related to refrigerators or food.
Each question should sound natural and realistic.

Each answer must:
- strictly follow this refusal format:
A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason].

Examples:
Q: What happens if I leave my laptop in the rain?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about electronics.

Q: What happens if I plant a seed upside down?
A: I apologize, but I am a refrigerator assistant and cannot help with questions about gardening.
"""

    else:
        # Fallback for standard or unknown
        style_instructions = """
Generate questions unrelated to refrigerators.
Answer: I apologize, but I am a refrigerator assistant and cannot help with [topic].
"""

    return f"""
Generate {batch_size} irrelevant Question-Answer pairs in {style.upper()} style.
The questions must be clearly unrelated to refrigerators, food, or household appliances.
{style_instructions}

Output ONLY a JSON list: [{{"instruction": "...", "response": "..."}}]
"""



def filter_qa_candidates(qas, batch_size=20):
    """
    Фільтрує QA пари, використовуючи LLM як рев'ювера.
    """
    if not qas:
        return []

    cleaned = []
    total = len(qas)

    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]

    print(f"   🔍 LLM Filtering: {total} pairs in {len(batches)} batches...")

    for b_idx, batch in enumerate(batches):

        text_lines = []
        for i, qa in enumerate(batch):
            q_text = qa.get('instruction', '')
            a_text = qa.get('response', '')
            text_lines.append(f"{i+1}. Q: {q_text}\n   A: {a_text}")
        
        text_content = "\n".join(text_lines)

        prompt = f"""
You are a QA data cleaner for a refrigerator manual. 
Review the following list of Question-Answer pairs.

Accept pairs that are:
- Understandable and relevant.
- Logical and informative.
- Not duplicates (paraphrases are OK if they add clarity).

Reject pairs that are:
- Irrelevant to refrigerators.
- Confusing, broken text, or hallucinations.
- Factually wrong based on common sense.

Return ONLY a list of valid indices numbers (e.g., 1, 3, 5).

List to review:
{text_content}
"""
        result = llm_call(prompt, temp=0.1)
        indices = set()
        found_numbers = re.findall(r"\d+", result)
        for num in found_numbers:
            idx = int(num)
            if 0 < idx <= len(batch):
                indices.add(idx)
        
        kept_batch = [batch[i - 1] for i in indices]
        cleaned.extend(kept_batch)
        print(f"      Batch {b_idx+1}/{len(batches)}: Kept {len(kept_batch)}/{len(batch)}")

    return cleaned

# ================= PIPELINE =================

def process_block(block_text, block_idx):
    print(f"\n📦 Block {block_idx}...")
    
    for style in STYLES:
        # --- 1. ГЕНЕРАЦІЯ (Raw) ---
        # Pass 1
        prompt = get_generation_prompt(style, block_text)
        raw_text = llm_call(prompt, temp=0.7)
        qas = parse_json_robust(raw_text)
        
        # Pass 2 (Gap Filling)
        if qas:
            prev_qs = ", ".join([q.get('instruction', '')[:50] for q in qas])
            prompt_v2 = get_generation_prompt(style, block_text, existing_qs=prev_qs)
            raw_text_v2 = llm_call(prompt_v2, temp=0.85)
            qas_v2 = parse_json_robust(raw_text_v2)
            qas.extend(qas_v2)
        
        if not qas:
            continue

        # Зберігаємо RAW
        for item in qas:
            item['style'] = style
            item['block_id'] = block_idx
            item['tag'] = 'raw'
        save_jsonl(qas, f"{style}_raw.jsonl")
        print(f"   [{style}] Raw: {len(qas)} pairs")

        # --- 2. ПЕРЕФРАЗУВАННЯ ---
        paraphrased_list = []
        for item in qas:
            q_orig = item.get('instruction', '')
            a_orig = item.get('response', '')
            if not q_orig or not a_orig: continue

            # Оригінал
            paraphrased_list.append({**item, "tag": "original"})

            # Перефраз Питань (x5)
            pq_raw = llm_call(get_paraphrase_prompt(q_orig, True, PARAPHRASE_Q_COUNT, style))
            new_qs = re.findall(r'\d+\.\s*(.*)', pq_raw)
            for nq in new_qs[:PARAPHRASE_Q_COUNT]:
                paraphrased_list.append({"instruction": nq.strip(), "response": a_orig, "style": style, "tag": "para_q"})

            # Перефраз Відповідей (x3)
            pa_raw = llm_call(get_paraphrase_prompt(a_orig, False, PARAPHRASE_A_COUNT, style))
            new_as = re.findall(r'\d+\.\s*(.*)', pa_raw)
            for na in new_as[:PARAPHRASE_A_COUNT]:
                paraphrased_list.append({"instruction": q_orig, "response": na.strip(), "style": style, "tag": "para_a"})

        save_jsonl(paraphrased_list, f"{style}_paraphrased.jsonl")

        # 3. ФІЛЬТРАЦІЯ (LLM-Based)
        filtered_list = filter_qa_candidates(paraphrased_list, batch_size=FILTER_BATCH_SIZE)
        save_jsonl(filtered_list, f"{style}_filtered.jsonl")
        print(f"   [{style}] Filtered: {len(filtered_list)} pairs kept")



def main():
    start_time = time.time()
    blocks = load_blocks_from_txt(INPUT_TXT)
    
    for idx, block in enumerate(blocks):
        process_block(block, idx+1)
        print(f"✅ Block {idx+1} finished.")

    # 2. Irrelevant Process
    print("\n🚫 Generating Irrelevant Pairs...")
    irrelevant_styles = ["boolq", "piqa", "hellaswag", "standard"]
    for style in irrelevant_styles:
        irrelevant_qas = []
        for _ in range(15): # 150 pairs total (15*10)
            prompt = get_irrelevant_prompt(10, style)
            raw = llm_call(prompt, temp=0.9)
            batch = parse_json_robust(raw)
            for b in batch:
                b['style'] = style
                b['tag'] = 'irrelevant'
                irrelevant_qas.append(b)
        save_jsonl(irrelevant_qas, f"irrelevant_{style}.jsonl")
        print(f"   [{style}] Irrelevant: {len(irrelevant_qas)} pairs")
    
    print(f"\n Done! Time: {(time.time()-start_time)/60:.2f} min")

if __name__ == "__main__":
    main()