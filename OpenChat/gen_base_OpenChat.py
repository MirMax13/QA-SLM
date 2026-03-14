import fitz  # PyMuPDF
import json
import re
from time import sleep
from dotenv import load_dotenv
import os

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

from OpenChat.common import num_tokens, call_lm, load_blocks_from_txt, parse_qa_pairs, save_qa, filter_qa_candidates
    
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
    with open(INSTRUCTION_PATH, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks):
            f.write(f"🔹 Блок {i+1} ({len(block.split())} слів):\n")
            f.write("-" * 60 + "\n")
            f.write(block + "\n")
            f.write("-" * 60 + "\n\n")

    return blocks

# ========== STEP 2: QA Generation ==========
def generate_qa_pairs(block_text):
    prompt = f"""
    <|im_start|>system
    You are an intelligent refrigerator that answers user questions related to the provided instruction text.
    Generate 2–3 natural and informative question–answer pairs based on the following manual section.
    If there is an opportunity to ask more questions and answers, this is encouraged (up to 5 pairs)
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
    import random
    topics = [
    "astronomy", "marine biology", "quantum physics", "world geography", "ancient history",
    "classical music", "modern art", "psychology", "economics", "literature",
    "dog training", "space exploration", "artificial intelligence", "programming languages", "automotive engineering",
    "renewable energy", "fashion design", "gardening", "fitness and exercise", "yoga",
    "martial arts", "basketball", "soccer", "tennis", "swimming",
    "video games", "board games", "chess", "photography", "architecture",
    "poetry", "theater and acting", "comic books", "mythology", "world religions",
    "philosophy", "sociology", "politics", "law and justice", "human anatomy",
    "genetics", "botany", "zoology", "ecology and environment", "weather and climate",
    "geology", "internet history", "smartphones", "aviation", "cybersecurity",
    "cryptocurrency", "virtual reality", "movies and cinema", "television shows", "rock music",
    "pop culture", "painting", "sculpture", "anime and manga", "World War II",
    "medieval Europe", "ancient Egypt", "Roman Empire", "languages and linguistics", "personal finance",
    "investing", "real estate", "taxes", "travel and tourism", "hotel management",
    "car maintenance", "home renovation", "history of the Olympics", "woodworking", "magic tricks",
    "folklore and legends", "astrology", "cryptography", "journalism", "marketing",
    "social media", "e-commerce", "pet care", "cat behavior", "aquarium keeping",
    "archaeology", "pharmacology", "space travel", "dinosaurs and paleontology", "evolution",
    "sailing and boating", "mountain climbing", "scuba diving", "origami", "playing the piano",
    "playing the guitar", "public speaking", "time management", "mental health", "mathematics"
    ]
    for b_idx, (start, end) in enumerate(batches):
        
        count = end - start
        current_topics = random.sample(topics, 3)
        topics_str = ", ".join(current_topics)
        print(f"🔄 Generating irrelevant QAs batch {b_idx+1}/{len(batches)} ({count} pairs)({count} pairs) - Topics: {topics_str}")
        
        prompt = f"""
        <|im_start|>system
        You are a QA data generator. 
        Write {count} unrelated questions that a refrigerator cannot answer, and assign each an appropriate refusal response.
        The questions must be clearly unrelated to refrigerators, food, or household appliances.
        CRITICAL: For this specific batch, you MUST strictly base your questions ONLY on these topics: {topics_str}. 
        Do not use generic trivia!

        
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
        save_qa(batch_qas, file_path=f"irrelevant_basic.json")
        qas.extend(batch_qas)
        
        print(f"✅ Generated {len(batch_qas)} pairs in batch {b_idx+1}")
        sleep(2)  # Add small delay between batches
    
    return qas[:n]  # Ensure we return exactly n pairs

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
