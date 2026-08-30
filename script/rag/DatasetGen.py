import json
import random

# 1. Функція для завантаження блоків з файлу
def load_chunks_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Розбиваємо текст по маркеру
    raw_blocks = content.split("🔹 Блок")[1:]
    
    chunks = []
    for block in raw_blocks:
        parts = block.split("-" * 60)
        if len(parts) >= 3:
            chunks.append(parts[1].strip())
    return chunks

# Завантажуємо наші блоки з інструкції
chunks = load_chunks_from_file("input/Instructions/Instruction_v1.4.txt")

updated_dataset = []

# 2. Читаємо твій старий датасет
input_filename = "datasets\\gpt-oss20b\\Full.jsonl"  # Назва твого поточного файлу
output_filename = "rag_dataset.jsonl" # Файл, який ми отримаємо на виході

with open(input_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        original_question = data["instruction"]
        
        # 1. Якщо це нерелевантне питання — даємо випадковий контекст
        # При цьому ми НЕ змінюємо current_block_id
        if data.get("tag") == "irrelevant":
            context = random.choice(chunks)
            
        else:
            # 2. Якщо в поточній парі є block_id, оновлюємо нашу "пам'ять"
            if "block_id" in data:
                current_block_id = data["block_id"]
            
            # 3. Витягуємо контекст за збереженим у пам'яті ID
            if current_block_id is not None:
                block_idx = current_block_id - 1
                
                # Запобіжник виходу за межі масиву
                if 0 <= block_idx < len(chunks):
                    context = chunks[block_idx]
                else:
                    context = random.choice(chunks)
            else:
                # На випадок, якщо найперший рядок у файлі не має block_id
                context = chunks[0]
                
        # 4. Формуємо об'єднаний промпт
        new_instruction = f"Context: {context}\n\nQuestion: {original_question}"
        data["instruction"] = new_instruction
        
        updated_dataset.append(data)

# 4. Зберігаємо оновлений датасет
with open(output_filename, "w", encoding="utf-8") as outfile:
    for item in updated_dataset:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Успішно! Оброблено {len(updated_dataset)} пар.")
print(f"Новий датасет збережено як {output_filename}.")