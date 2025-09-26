import json

# Таблиця замін битих символів
fix_map = {
    "â€™": "'",
    "â€œ": "“",
    "â€": "”",
    "Ã©": "é",   # інколи з французькими літерами теж буває
}

def fix_text(s: str) -> str:
    for bad, good in fix_map.items():
        s = s.replace(bad, good)
    return s

input_file = "filtered_tinystories/tiny_stories_train_filtered_20.json"
output_file = "filtered_tinystories/tiny_stories_train_20.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Проходимо всі приклади й виправляємо тексти
for item in data:
    if "text" in item:
        item["text"] = fix_text(item["text"])

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Записано у {output_file}")
