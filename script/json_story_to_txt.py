import json

# Вхідний JSON
input_file = "./results/GPT/ChatGpt/TinyStories/model_40mb_results_last.json"   # твій файл із JSON
output_file = "./results/GPT/ChatGpt/TinyStories/model_40mb_results_last.txt"   # файл куди запишемо TXT

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(data, 1):
        f.write(f"### Example {i}\n")
        f.write("Input:\n")
        f.write(item["input_text"].strip() + "\n\n")
        f.write("Generated:\n")
        f.write(item["generated_story"].strip() + "\n")
        f.write("-" * 40 + "\n\n")

print(f"Готово ✅ Записано у {output_file}")
