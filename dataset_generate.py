import fitz  # PyMuPDF
import requests
import json
import re
from time import sleep

PDF_PATH = "DA68-04844A-00_EN.pdf"
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "openchat"
OUTPUT_JSON = "fridge_dataset_from_pdf.json"

# 1. Витягуємо текст з PDF
def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    # Розбиваємо по заголовках або великих блоках (можеш налаштувати)
    blocks = re.split(r'\n(?=[A-ZА-Я][^\n]{0,80}\n)', text)  # новий блок починається з заголовка
    for i, block in enumerate(blocks[:5]):
        print(f"\n🔹 Блок {i+1}:\n{block[:300]}...\n{'-'*50}")
    return [b.strip() for b in blocks if len(b.strip()) > 100]

# 2. Генеруємо питання-відповіді для кожного блоку
def generate_qa(block_text):
    prompt = f"""
    <|im_start|>system
    Ти — інтелектуальний холодильник, що відповідає на запитання користувача.
    Згенеруй 2 запитання, які можуть стосуватися наступного тексту інструкції, і дай до кожного повну відповідь.
    Формат:
    Q: ...
    A: ...
    Q: ...
    A: ...
    Текст інструкції:
    \"\"\"{block_text}\"\"\"
    <|im_end|>
    <|im_start|>user
    Створи QA пари<|im_end|>
    <|im_start|>assistant
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(LM_API_URL, headers=HEADERS, json=payload, timeout=40)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_qa_pairs(content)
    except Exception as e:
        print(f"❌ Помилка генерації: {e}")
        return []

# 3. Парсимо результат
def parse_qa_pairs(text):
    qas = []
    qa_blocks = re.findall(r"Q:(.*?)A:(.*?)(?=Q:|$)", text, re.DOTALL)
    for q, a in qa_blocks:
        question = q.strip().replace("\n", " ")
        answer = a.strip().replace("\n", " ")
        if question and answer:
            qas.append({
                "instruction": question,
                "response": answer
            })
    return qas

# 4. Основний процес
def main():
    blocks = extract_text_blocks(PDF_PATH)
    dataset = []

    for idx, block in enumerate(blocks):
        print(f"🔹 Обробляємо блок {idx+1}/{len(blocks)}")
        qa_pairs = generate_qa(block)
        dataset.extend(qa_pairs)

        for qa in qa_pairs:
            print(f"Q: {qa['instruction']}")
            print(f"A: {qa['response']}\n---")
        sleep(2)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Згенеровано {len(dataset)} QA пар, збережено у {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
