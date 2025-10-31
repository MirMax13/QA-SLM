import json
import re
from time import sleep
from dotenv import load_dotenv
from config.config import INSTRUCTION_PATH, OUTPUT_JSON, OUTPUT_JSON_CLEANED

load_dotenv()

mode = "hellaswag"  # boolq / piqa / hellaswag

# shared helpers
from OpenChat.common import num_tokens, call_lm, load_blocks_from_txt, parse_qa_pairs, save_qa, filter_qa_candidates

# ========== STEP 2: QA Generation ==========
def generate_qa_pairs(block_text, mode="boolq"):
    if mode == "boolq":
        prompt = f"""
<|im_start|>system
You are an intelligent refrigerator that gives helpful, safety-conscious, and detailed answers.
Generate 3–5 natural question–answer pairs inspired by the BoolQ format.
The user asks if something can or should be done, and you answer "Yes" or "No" with reasoning.
Answers must be realistic, safety-aware, and may include alternatives or extra advice.
Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.

Examples:
Q: Can I store eggs in the freezer?
A: No, you shouldn’t store eggs in the freezer, because freezing damages their texture and shell. It's better to keep them on a refrigerator shelf instead.

Q: Is it safe to put hot soup directly into the fridge?
A: No, it's not safe. Placing hot food directly can raise the temperature inside the fridge, affecting other products.

Q: Can I clean the refrigerator with bleach?
A: Yes, but only in diluted form and with proper ventilation. Avoid direct contact with food surfaces.

The text of the instructions:
\"\"\"{block_text}\"\"\"
    <|im_end|>
    <|im_start|>user
    Generate QA pairs<|im_end|>
    <|im_start|>assistant
    """
    elif mode == "piqa":
        prompt = f"""
<|im_start|>system
You are an intelligent refrigerator that helps users make the best practical decisions in everyday home situations.
Generate 3–5 natural PIQA-style question–answer pairs related to the text below.

Each question must:
- describe a small realistic situation involving refrigerators or appliance care
- **always include 2–3 possible actions or options** that a person might choose (some may be wrong)
- avoid "How can I..." or "What should I..." phrasing — use "Is it better to...", "Should I...", "Would it make sense to..." etc.
- never use markers like A/B/C or numbers
- be phrased conversationally and concisely (1 sentence)

Each answer must:
- clearly state which option is correct (or that neither is good)
- explain *why* that choice is best, with short reasoning
- be friendly, helpful, and realistic

Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.

Examples:
Q: Where is it better to store eggs — in the freezer or on the refrigerator shelf?
A: It’s better to store eggs on the refrigerator shelf where the temperature is stable, not in the freezer where they may crack.

Q: Should I place vegetables in the main compartment or in the crisper drawer?
A: Vegetables should go in the crisper drawer — it keeps higher humidity and helps them stay fresh longer.

Q: Is it better to clean the fridge with oil or milk?
A: Neither option is suitable. Oil can leave greasy residue, and milk can spoil quickly. Baking soda is a better choice — it absorbs odors and is gentle on the surfaces.


The text of the instructions:
\"\"\"{block_text}\"\"\"
<|im_end|>
<|im_start|>user
Generate PIQA-style QA pairs<|im_end|>
<|im_start|>assistant
"""

    elif mode == "hellaswag":
        prompt = f"""
<|im_start|>system
You are an intelligent refrigerator that explains the results of user actions.
Generate 3–5 natural question–answer pairs inspired by the HellaSwag format.
Each question should describe an action or event (“What happens if...”, “What will occur when...”), and the answer should describe the likely outcome and reason.
Make the answers as complete, helpful, and context-aware as possible.
Avoid overly short or generic answers. Even if the core answer is simple, elaborate on the reasoning, details, or implications to ensure helpfulness.

Examples:
Q: What happens if I leave the fridge door open overnight?
A: The temperature inside will rise, food may spoil, and frost can form when the compressor tries to cool down again.

Q: What happens if I store raw meat above vegetables?
A: The juices from the meat could drip down and contaminate the vegetables, increasing the risk of foodborne illness.

Q: What happens if I turn off the fridge but keep food inside?
A: The temperature will rise within a few hours, and perishable foods like dairy or meat will spoil quickly.

The text of the instructions:
\"\"\"{block_text}\"\"\"
<|im_end|>
<|im_start|>user
Generate HellaSwag-style QA pairs<|im_end|>
<|im_start|>assistant
"""

    if len(prompt) > 12000:
        print("⚠️ Prompt too long")
    content = call_lm([{"role": "user", "content": prompt}])
    return parse_qa_pairs(content)

# ========== STEP 4: Paraphrasing ==========
def generate_paraphrases(text, is_question=True, format_type=None, n=3):
    role = "question" if is_question else "answer"
    style_hint = ""
    if format_type == "boolq":
        style_hint = "Preserve the 'Yes/No with reasoning' style and tone."
    elif format_type == "piqa":
        style_hint = "Keep the comparative structure, mentioning multiple options naturally."
    elif format_type == "hellaswag":
        style_hint = "Preserve the 'what happens if' reasoning tone."
        
    prompt = f"""
<|im_start|>system
You are a helpful assistant.
Generate {n} diverse paraphrases of the following {role}, preserving its meaning.
Do NOT number or label the paraphrases.
Do NOT add quotes or special formatting.
{style_hint}
{role.capitalize()}: "{text}"
<|im_end|>
<|im_start|>user
Paraphrase<|im_end|>
<|im_start|>assistant
"""
    raw = call_lm([{"role": "user", "content": prompt}])
    lines = [re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', l.strip("-• ")) for l in raw.strip().splitlines() if l.strip()]
    return lines[:n]


# ========== STEP 5: Irrelevant QA ==========

def generate_irrelevant_qas(mode: str, n=50, batch_size=10):
    assert mode in ["boolq", "piqa", "hellaswag"], "mode must be one of: boolq, piqa, hellaswag"
    
    qas = []
    batches = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    
    for b_idx, (start, end) in enumerate(batches):
        count = end - start
        print(f"🔄 Generating irrelevant {mode.upper()} QAs batch {b_idx+1}/{len(batches)} ({count} pairs)")
        
        # === Base prompt, depending on the selected mode ===
        if mode == "boolq":
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

        elif mode == "piqa":
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

        elif mode == "hellaswag":
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

        # === Compose full prompt ===
        prompt = f"""
<|im_start|>system
You are a QA generator creating *irrelevant* question–answer pairs in {mode.upper()} style.
The questions must be clearly unrelated to refrigerators, food, or household appliances.
Each answer must be a polite refusal in the exact format shown below.
Do NOT answer the question content — only refuse.
Generate {count} pairs.

Use exactly this format:
Q1: [question]
A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason].
Q2: [question]
A: I apologize, but I am a refrigerator assistant and cannot help with [topic-specific reason].

{style_instructions}
<|im_end|>
<|im_start|>user
Generate unrelated {mode.upper()}-style QA pairs.<|im_end|>
<|im_start|>assistant
"""
        
        if num_tokens(prompt) > 8000:
            print(f"⚠️ Generation prompt too long in batch {b_idx+1}")
            
        text = call_lm([{"role": "user", "content": prompt}])
        batch_qas = parse_qa_pairs(text)
        save_qa(batch_qas, file_path=f"irrelevant_{mode}.json")
        qas.extend(batch_qas)
        
        print(f"✅ Generated {len(batch_qas)} pairs in batch {b_idx+1}")
        sleep(2)  # Add small delay between batches
    
    return qas[:n]  # Ensure we return exactly n pairs

# ========== STEP 6: Main loop ==========
def main():
    # blocks = extract_text_blocks(PDF_PATH)
    print(INSTRUCTION_PATH)
    blocks = load_blocks_from_txt(INSTRUCTION_PATH)
    dataset = []
    dataset_cleaned = []
    for idx, block in enumerate(blocks):
        print(f"\n🔷 Processing block {idx+1}/{len(blocks)}")
        base_qas = generate_qa_pairs(block, mode)
        all_qas = []
        # Save base QA in json
        for qa in base_qas:
            save_qa(qa, file_path=f"orig_{mode}.json")

        for qa in base_qas:
            paraphrased_qs = generate_paraphrases(qa['instruction'], is_question=True, n=5)
            paraphrased_as = generate_paraphrases(qa['response'], is_question=False, n=3)

            for pq in [qa['instruction']] + paraphrased_qs:
                for pa in [qa['response']] + paraphrased_as:
                    all_qas.append({"instruction": pq, "response": pa, "tag": "good"})

        dataset.extend(all_qas)
        # Save original + paraphrased in json
        for qa in all_qas:
            save_qa(qa, file_path=f"paraphrased_{mode}.json")

        filtered_qas = filter_qa_candidates(all_qas)
        for qa in filtered_qas:
            save_qa(qa, file_path=f"filtered_{mode}.json")
        dataset_cleaned.extend(filtered_qas)
        print(f"➕ Added {len(all_qas)} QA pairs from block {idx+1}")
        print(f"✅ {len(filtered_qas)} kept after filtering")
        sleep(2)
    
    # Add irrelevant QAs
    print("\n🚫 Generating irrelevant questions...")
    irrelevant_qas = generate_irrelevant_qas(mode, n=150, batch_size=10)
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
