import os
import json
import re
import time
import torch
from transformers import pipeline
import ast

INPUT_TXT = "Instruction_v1.4.txt"
MODEL_ID = "models/openai_gpt-oss-20b"
OUTPUT_DIR = "output"

PARAPHRASE_Q_COUNT = 3  
PARAPHRASE_A_COUNT = 2  
STYLES = ["hellaswag"]
FILTER_BATCH_SIZE = 15
CYCLES = 1
BATCHES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Initializing model...")
try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def load_blocks_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_blocks = re.split(r'🔹 Блок \d+ \(\d+ слів\):\n[-]+\n(.*?)\n[-]+\n', content, flags=re.DOTALL)
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    print(f"📄 Завантажено {len(blocks)} блоків із {file_path}")
    return blocks

def save_jsonl(entries, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not isinstance(entries, list):
        entries = [entries]
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def llm_call(messages_list, max_new=2048, temp=0.3):

    prompt = ""
    for msg in messages_list:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            return_full_text=False
        )
        text = outputs[0]["generated_text"].strip()
        # === DEBUG ===
        print(f"\n🔎 DEBUG RAW OUTPUT:\n{text}...\n")
        return text
    except Exception as e:
        print(f"⚠️ Generation Error: {e}")
        return ""

def extract_json_from_markdown(text):
    text = text.replace("assistantfinal", "") 

    # Знаходимо всі дужки (і квадратні, і фігурні)
    starts = [m.start() for m in re.finditer(r'[\[\{]', text)]
    ends = [m.start() for m in re.finditer(r'[\]\}]', text)]
    
    if not starts or not ends:
        return []
        
    # Шукаємо З КІНЦЯ
    for end in reversed(ends):
        # Шукаємо відповідний початок
        valid_starts = [s for s in starts if s < end]
        
        for start in reversed(valid_starts):
            candidate = text[start : end+1]
            
            # Базова перевірка балансу (не гарантія, але відсіює явне сміття)
            # Якщо почали з {, маємо закінчити }. Якщо з [, то ].
            first_char = candidate[0]
            last_char = candidate[-1]
            if (first_char == '{' and last_char != '}') or (first_char == '[' and last_char != ']'):
                continue

            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass
                
    return []
def get_messages(style, text, existing_qs=""):
    system_content = (
        "You are a dataset generator. Your ONLY goal is to output a valid JSON array inside a ```json``` markdown block.\n"
        "TONE RULES: Simulate a real user asking for help. Use simple, conversational English. "
        "Avoid formal jargon (e.g., say 'Is it better to...' instead of 'Does the manual mandate...').\n"
        "CRITICAL: Do NOT provide explanations, reasoning, or intros. Start your response IMMEDIATELY with ```json."
    )
    
    avoid_instr = f"Do NOT generate these questions again: {existing_qs}." if existing_qs else ""
    
    user_content = ""
    if style == "standard":
        user_content = f"""
Based on the text, generate 5 Question-Answer pairs.
Text: \"\"\"{text}\"\"\"

Output format example (Do not copy this, generate new based on text):
```json
[
  {{"instruction": "What should I do before cleaning?", "response": "Unplug the power cord to avoid electric shock."}},
  {{"instruction": "Where is the water filter located?", "response": "It is located in the bottom right corner of the fridge."}}
]
{avoid_instr}
Output:
"""
    elif style == "boolq":
        user_content = f"""
Generate 5 'Yes/No' questions based on the text. Answer with 'Yes/No' + reasoning.
Text: \"\"\"{text}\"\"\"

Output format example:
```json
[
  {{"instruction": "Can I use abrasive cleaners?", "response": "No, because they can scratch the surface."}},
  {{"instruction": "Is the door reversible?", "response": "Yes, the door can be installed to open from either side."}}
]
{avoid_instr}
Output:
"""
    elif style == "piqa":
        user_content = f"""
Generate 5 comparison questions (Option A vs B) based on the text.
Text: \"\"\"{text}\"\"\"

Output format example:
```json
[
  {{"instruction": "Is it beneficial to leave the doors of my refrigerator open for a short time after putting in new groceries, or is immediate closure more appropriate?", "response": "For optimal preservation of your groceries, it's advisable to promptly shut the refrigerator doors following their addition. This practice ensures that the interior temperature remains stable, thereby reducing the rate at which your food can spoil."}},
  {{"instruction": "To obtain the best results, how frequently should one utilize the Power Freeze feature?", "response": "Consider employing Power Freeze on a regular basis, such as every few days, to rapidly freeze your items; however, make sure to revert the freezer to its initial temperature setting afterward. Using it too frequently may lead to higher energy usage."}}
]
{avoid_instr}
Output:
"""
    elif style == "hellaswag":
        user_content = f"""
Generate 5 'What happens if...' questions based on the text.
Text: \"\"\"{text}\"\"\"

Output format example:
```json
[
  {{"instruction": "Can one use baking soda for cleaning the insides of a refrigerator?", "response": "Baking soda should not be used for cleaning, as it might damage the refrigerator's surface and cause a fire. Follow the manufacturer's recommended guidelines for safe cleaning procedures."}},
  {{"instruction": "What happens if I accidentally reverse the order of shelves in a Type B refrigerator?", "response": "The layout of the shelf will be altered, which could impact your organization and ease of accessing products. In a Type B refrigerator, rearranging the shelves might lead to suboptimal use of space and challenges in finding items."}}
]
{avoid_instr}
Output:
"""
    return [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]

    
def get_paraphrase_messages(text, is_question, count, style="standard"):
    role = "question" if is_question else "answer"
    user_content = f"""
Rewrite the following {role} in {count} different ways.
Original: "{text}"

RULES:
1. Output ONLY a numbered list.
2. NO explanations. NO conversational filler.
3. Start immediately with "1. ".

Output:
"""
    return [{"role": "user", "content": user_content}]

def get_irrelevant_messages(batch_size, style):
    system_content = "You are a dataset generator. Output JSON inside json block only."
    user_content = f""" Generate {batch_size} questions completely UNRELATED to refrigerators (e.g. history, space).
      Style: {style.upper()}. Refusal Answer MUST be: "I apologize, but I am a refrigerator assistant and cannot help with [topic]."
        Output format:
        ```json
        [
  {{"instruction": "Question...", "response": "Refusal..."}}
]
    """
    return [ {"role": "system", "content": system_content}, {"role": "user", "content": user_content} ]

def filter_qa_candidates(qas, batch_size=25):
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
            text_lines.append(f"ID {i+1}:\nQ: {q_text}\nA: {a_text}\n")

        text_content = "\n".join(text_lines)

        prompt_text = f"""
You are a dataset quality checker. We are building a dataset for training AI, so we need MANY different ways to ask the same question.

INPUT LIST:
{text_content}

CRITERIA FOR KEEPING (Valid IDs):
1. **High Quality**: The text is complete, grammatical, and makes sense.
2. **Paraphrases are WANTED**: If two items ask the same thing but use DIFFERENT words, KEEP BOTH. This is data augmentation.
   - Example: "How do I clean it?" and "What is the cleaning procedure?" -> KEEP BOTH.

CRITERIA FOR REJECTING (Exclude these IDs):
1. **Garbage**: Text that is cut off, incomplete, or contains meta-instructions (e.g., "Here is the rewrite").
2. **Exact Duplicates**: Only reject if the wording is 100% identical to a previous item in this batch.

TASK:
Output a JSON object with a single key "valid_ids" containing the list of ID numbers to keep.
Example: {{ "valid_ids": [1, 2, 4, 5] }}
NO explanations. Just the JSON. """

        messages = [{"role": "user", "content": prompt_text}]
        result = llm_call(messages, temp=0.2)
        parsed_json = extract_json_from_markdown(result)
    
        valid_indices = []
        if isinstance(parsed_json, dict) and "valid_ids" in parsed_json:
            valid_indices = parsed_json["valid_ids"]
        elif isinstance(parsed_json, list): 
            valid_indices = parsed_json
            
        if not valid_indices:
            print(f"      ⚠️ Warning: Could not parse filter decision. Keeping all {len(batch)} for safety.")
            valid_indices = range(1, len(batch) + 1)

        indices_set = set(valid_indices)
        kept_batch = []
        for i in range(len(batch)):
            if (i + 1) in indices_set:
                kept_batch.append(batch[i])
        
        cleaned.extend(kept_batch)
        print(f"      Batch {b_idx+1}/{len(batches)}: Kept {len(kept_batch)}/{len(batch)}")

    return cleaned

# ================= PIPELINE =================

def process_block(block_text, block_idx):
    print(f"\n📦 Block {block_idx}...")
    
    for style in STYLES:
        print(f"Checking style: {style}...")
        
        # === 1. СПРОБА ЗАВАНТАЖИТИ ІСНУЮЧІ ДАНІ ===
        raw_filename = f"{style}_raw.jsonl"
        raw_path = os.path.join(OUTPUT_DIR, raw_filename)
        qas = []

        if os.path.exists(raw_path):
            print(f"   Found existing file: {raw_filename}. Reading...")
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            # Беремо тільки ті, що належать поточному блоку
                            if entry.get('block_id') == block_idx:
                                qas.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"   ⚠️ Error reading file: {e}")

        if qas:
            print(f"   ✅ Loaded {len(qas)} pairs from file. Skipping generation.")
        else:
            print("   Generating new Q&A pairs (LLM)...")
            prompt = get_messages(style, block_text)
            raw_text = llm_call(prompt, temp=0.3)
            
            print("   Parsing JSON array...")
            qas = extract_json_from_markdown(raw_text)

            # Gap Filling
            if qas:
                print("   Gap Filling...")
                # Формуємо список існуючих питань для промпта
                prev_qs = ", ".join([q.get('instruction', '')[:50] for q in qas])
                
                prompt_v2 = get_messages(style, block_text, existing_qs=prev_qs)
                raw_text_v2 = llm_call(prompt_v2, temp=0.4)
                qas_v2 = extract_json_from_markdown(raw_text_v2)


                existing_instructions = set(item.get('instruction', '').strip().lower() for item in qas)

                unique_new_count = 0
                for new_item in qas_v2:
                    new_q = new_item.get('instruction', '').strip().lower()
                    
                    # Додаємо тільки якщо питання не пусте і його ще немає в списку
                    if new_q and new_q not in existing_instructions:
                        qas.append(new_item)
                        existing_instructions.add(new_q)
                
                print(f"   Gap Filling added {unique_new_count} unique pairs (ignored {len(qas_v2) - unique_new_count} duplicates)")
                # ============================================

            if not qas:
                print(f"   [{style}] Empty or invalid JSON from model.")
                continue

        # Зберігаємо RAW
        for item in qas:
            item['style'] = style
            item['block_id'] = block_idx
            item['tag'] = 'raw'
        save_jsonl(qas, f"{style}_raw.jsonl")
        print(f"   [{style}] Raw: {len(qas)} pairs")

        paraphrased_list = []
        for item in qas:
            q_orig = item.get('instruction', '')
            a_orig = item.get('response', '')
            if not q_orig or not a_orig: continue

            paraphrased_list.append({**item, "tag": "original"})

            pq_raw = llm_call(get_paraphrase_messages(q_orig, True, PARAPHRASE_Q_COUNT, style), temp=0.5)
            
            pq_raw = pq_raw.replace("assistantfinal", "")

            new_qs = re.findall(r'^\d+\.\s+(.{5,})$', pq_raw, re.MULTILINE)

            clean_qs = []
            for q in new_qs:
                q = q.strip('" ').strip()

                lower_q = q.lower()
                if (lower_q.startswith("we need to") or 
                    lower_q.startswith("here is") or 
                    lower_q.startswith("the rewritten") or
                    len(q) > len(q_orig) * 3):
                    continue
                clean_qs.append(q)
            
            for nq in clean_qs[:PARAPHRASE_Q_COUNT]:
                paraphrased_list.append({"instruction": nq.strip(), "response": a_orig, "style": style, "tag": "para_q"})

            pa_raw = llm_call(get_paraphrase_messages(a_orig, False, PARAPHRASE_A_COUNT, style), temp=0.5)
            pa_raw = pa_raw.replace("assistantfinal", "")
            
            new_as = re.findall(r'^\d+\.\s+(.{5,})$', pa_raw, re.MULTILINE)
            
            clean_as = []
            for a in new_as:
                a = a.strip('" ').strip()
                lower_a = a.lower()
                if (lower_a.startswith("we need to") or 
                    lower_a.startswith("here is") or 
                    lower_a.startswith("i will rewrite") or
                    len(a) > len(a_orig) * 3):
                    continue
                clean_as.append(a)

            for na in clean_as[:PARAPHRASE_A_COUNT]:
                paraphrased_list.append({"instruction": q_orig, "response": na.strip(), "style": style, "tag": "para_a"})

        if paraphrased_list:
            save_jsonl(paraphrased_list, f"{style}_paraphrased.jsonl")

            filtered_list = filter_qa_candidates(paraphrased_list, batch_size=FILTER_BATCH_SIZE)
            save_jsonl(filtered_list, f"{style}_filtered.jsonl")
            print(f"   [{style}] Paraphrased & Saved: {len(paraphrased_list)} pairs")

def main():
    start_time = time.time()
    blocks = load_blocks_from_txt(INPUT_TXT)
    
    # 1. Main Process
    for idx, block in enumerate(blocks):
        if idx == 1:
            process_block(block, idx+1)
            print("🛑 STOPPING after 1 block for TESTING purposes.")
            break 
    
    # 2. Irrelevant Process
    print("\n🚫 Generating Irrelevant Pairs...")
    for style in STYLES:
        irrelevant_qas = []
        for _ in range(CYCLES): 
            raw = llm_call(get_irrelevant_messages(BATCHES, style), temp=0.5)
            batch = extract_json_from_markdown(raw)

            if isinstance(batch, dict):
                batch = [batch]
            
            if not isinstance(batch, list):
                continue

            for b in batch:
                if not isinstance(b, dict):
                    continue
                
                b['style'] = style
                b['tag'] = 'irrelevant'
                irrelevant_qas.append(b)
        
        if irrelevant_qas:
            save_jsonl(irrelevant_qas, f"irrelevant_{style}.jsonl")
            print(f"   [{style}] Irrelevant: {len(irrelevant_qas)} pairs")
        else:
            print(f"   [{style}] ⚠️ Failed to generate irrelevant pairs.")

    print(f"\nDone! Time: {(time.time()-start_time)/60:.2f} min")

if __name__ == "__main__":
    main()