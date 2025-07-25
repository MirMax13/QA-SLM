import json
import re

def find_answer_start(context, answer):
    # Якщо відповіді немає, повертаємо -1
    if not answer:
        return -1

    idx = context.find(answer)
    if idx != -1:
        return idx

    # Спроба з очищенням пробілів і лапок
    answer_clean = re.sub(r'\s+', ' ', answer).strip().strip('"\'')
    context_clean = re.sub(r'\s+', ' ', context)

    idx_clean = context_clean.find(answer_clean)
    if idx_clean != -1:
        print(f"🧪 Approximate match for: {answer_clean}")
        return idx_clean

    print(f"⚠️ Failed to find: {answer}")
    return -1

def update_answer_starts(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        context = entry["context"]
        
        # Перевірка на наявність відповіді
        if not entry["answers"]:
            continue  # Пропускаємо записи, де відповіді немає

        for ans in entry["answers"]:
            answer = ans["text"]
            ans["answer_start"] = find_answer_start(context, answer)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Updated file saved to {output_path}")

# 🔧 Запуск:
update_answer_starts("./datasets/ChatGPT/extractive/fridge_dataset_v1.1_clean.json", "res5.json")
