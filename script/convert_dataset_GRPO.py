import json
from config.config import OUTPUT_JSON_CLEANED_GEN
# Завантажуємо оригінальний датасет
with open(OUTPUT_JSON_CLEANED_GEN, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Створюємо нову структуру
grouped_data = {}

for entry in original_data:
    question = entry['instruction']
    response = entry['response']
    tag = entry['tag']  # Зберігаємо тег

    if question not in grouped_data:
        grouped_data[question] = {"responses": [], "tags": []}

    # Додаємо відповідь та тег до відповідного питання
    grouped_data[question]["responses"].append(response)
    grouped_data[question]["tags"].append(tag)

# Перетворюємо на потрібний формат
final_data = [{"instruction": question, "responses": responses} for question, responses in grouped_data.items()]

# Зберігаємо новий датасет у файл
with open('grouped_qas.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"Збережено {len(final_data)} запитів у файл 'grouped_qas.json'")
