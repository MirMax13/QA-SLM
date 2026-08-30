import json
import numpy as np
from transformers import BartTokenizer

# 1. Завантажуємо токенізатор, який ти будеш використовувати для навчання
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

instructions_tokens = []
responses_tokens = []

# 2. Читаємо наш новий RAG-датасет
with open("rag_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        
        # Токенізуємо текст (add_special_tokens=False, щоб рахувати тільки сам текст)
        inst_encoded = tokenizer.encode(data["instruction"], add_special_tokens=False)
        resp_encoded = tokenizer.encode(data["response"], add_special_tokens=False)
        
        instructions_tokens.append(len(inst_encoded))
        responses_tokens.append(len(resp_encoded))

# 3. Виводимо детальну статистику
print("📊 АНАЛІЗ INSTRUCTION (Context + Question)")
print("-" * 40)
print(f"Мінімальна довжина: {np.min(instructions_tokens)} токенів")
print(f"Максимальна довжина: {np.max(instructions_tokens)} токенів")
print(f"Середня довжина:    {int(np.mean(instructions_tokens))} токенів")
print(f"95-й перцентиль:    {int(np.percentile(instructions_tokens, 95))} токенів (95% текстів коротші за це значення)")
print()

print("📊 АНАЛІЗ RESPONSE (Відповідь)")
print("-" * 40)
print(f"Мінімальна довжина: {np.min(responses_tokens)} токенів")
print(f"Максимальна довжина: {np.max(responses_tokens)} токенів")
print(f"Середня довжина:    {int(np.mean(responses_tokens))} токенів")
print(f"95-й перцентиль:    {int(np.percentile(responses_tokens, 95))} токенів")