import json

with open("fridge_dataset_piqa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Є {len(data)} об’єктів.")

