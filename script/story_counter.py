import ijson

filename = "stories.json"
count = 0

with open(filename, "r", encoding="utf-8") as f:
    for _ in ijson.items(f, "item"):
        count += 1

print("Кількість історій:", count)
