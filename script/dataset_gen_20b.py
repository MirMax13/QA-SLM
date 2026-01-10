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

PARAPHRASE_Q_COUNT = 1  
PARAPHRASE_A_COUNT = 1  
STYLES = ["boolq"]
FILTER_BATCH_SIZE = 5
CYCLES = 1
BATCHES = 5

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
    """
    Robust extraction:
    1. Tries to find markdown blocks.
    2. Tries to find the raw list structure [ ... ].
    3. Tries json.loads (strict).
    4. Tries ast.literal_eval (permissive, handles trailing commas/single quotes).
    """
    candidates = []

    # 1. Regex to find markdown blocks (ignoring case, optional "json" tag)
    # This regex looks for ``` ... ``` containing a square bracket at the start
    pattern_block = r"```(?:json)?\s*(\[.*?\])\s*```"
    matches = re.findall(pattern_block, text, re.DOTALL | re.IGNORECASE)
    candidates.extend(matches)

    # 2. Fallback: Find the outermost brackets if no blocks found
    if not candidates:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            candidates.append(text[start:end+1])

    for json_str in candidates:
        # Attempt 1: Standard JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Attempt 2: AST literal_eval (Handles trailing commas and single quotes)
        try:
            # AST is safer than eval() but handles python-syntax dicts
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass
            
        # Attempt 3: Try to clean trailing commas manually (common LLM error)
        try:
            # Remove comma before closing bracket or brace
            cleaned_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            pass

    return []
def get_messages(style, text, existing_qs=""):
    system_content = (
        "You are a strict dataset generator. "
        "Your ONLY goal is to output a valid JSON array inside a ```json``` markdown block. "
        "Do NOT provide explanations, reasoning, or intros."
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
            text_lines.append(f"{i+1}. Q: {q_text}\n    A: {a_text}")

        text_content = "\n".join(text_lines)

        prompt_text = f"""
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
        messages = [{"role": "user", "content": prompt_text}]
        result = llm_call(messages, temp=0.1)
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
            # === 2. ЯКЩО ДАНИХ НЕМАЄ — ГЕНЕРУЄМО ===
            print("   Generating new Q&A pairs (LLM)...")
            prompt = get_messages(style, block_text)
            raw_text = llm_call(prompt, temp=0.3)
            
            print("   Parsing JSON array...")
            qas = extract_json_from_markdown(raw_text)

            # Gap Filling (Догенерація)
            if qas:
                print("   Gap Filling...")
                prev_qs = ", ".join([q.get('instruction', '')[:50] for q in qas])
                prompt_v2 = get_messages(style, block_text, existing_qs=prev_qs)
                raw_text_v2 = llm_call(prompt_v2, temp=0.4)
                qas_v2 = extract_json_from_markdown(raw_text_v2)
                qas.extend(qas_v2)

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
            new_qs = re.findall(r'^\d+\.\s+(.{5,})$', pq_raw, re.MULTILINE)
            new_qs = [q.replace("assistantfinal", "").strip('" ').strip() for q in new_qs]
            for nq in new_qs[:PARAPHRASE_Q_COUNT]:
                paraphrased_list.append({"instruction": nq.strip(), "response": a_orig, "style": style, "tag": "para_q"})

            pa_raw = llm_call(get_paraphrase_messages(a_orig, False, PARAPHRASE_A_COUNT, style), temp=0.5)
            new_as = re.findall(r'^\d+\.\s+(.{5,})$', pa_raw, re.MULTILINE)
            new_as = [a.replace("assistantfinal", "").strip('" ').strip() for a in new_as]
            for na in new_as[:PARAPHRASE_A_COUNT]:
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
    # print("\n🚫 Generating Irrelevant Pairs...")
    # for style in STYLES:
    #     irrelevant_qas = []
    #     for _ in range(CYCLES): 
    #         prompt = get_irrelevant_prompt(BATCHES, style)
    #         raw = llm_call(prompt, temp=0.5)
    #         batch = extract_json_array(raw)
    #         for b in batch:
    #             b['style'] = style
    #             b['tag'] = 'irrelevant'
    #             irrelevant_qas.append(b)
        
    #     if irrelevant_qas:
    #         save_jsonl(irrelevant_qas, f"irrelevant_{style}.jsonl")
    #         print(f"   [{style}] Irrelevant: {len(irrelevant_qas)} pairs")
    #     else:
    #         print(f"   [{style}] ⚠️ Failed to generate irrelevant pairs.")

    print(f"\nDone! Time: {(time.time()-start_time)/60:.2f} min")

if __name__ == "__main__":
    main()