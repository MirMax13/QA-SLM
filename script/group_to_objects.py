import json

with open("fridge_dataset_piqa_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

flattened = [item for sublist in data for item in sublist]

with open("fridge_dataset_piqa_cleaned2.json", "w", encoding="utf-8") as f:
    json.dump(flattened, f, ensure_ascii=False, indent=2)

print(f"{len(data)} groups -> {len(flattened)} objects.")
