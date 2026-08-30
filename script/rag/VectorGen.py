import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Функція для автоматичного парсингу файлу
def load_chunks_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Розбиваємо весь текст по маркеру "🔹 Блок"
    # Пропускаємо перший елемент [1:], бо до першого блоку текст порожній
    raw_blocks = content.split("🔹 Блок")[1:]
    
    chunks = []
    for block in raw_blocks:
        # У твоєму файлі текст лежить між двома лініями по 60 дефісів
        parts = block.split("-" * 60)
        
        # Якщо блок розбився коректно, беремо текст між дефісами (це індекс 1)
        if len(parts) >= 3:
            chunk_text = parts[1].strip()
            chunks.append(chunk_text)
            
    return chunks

# 2. Автоматично завантажуємо блоки
filename = "input/Instructions/Instruction_v1.4.txt"
chunks = load_chunks_from_file(filename)
print(f"Успішно завантажено блоків з файлу: {len(chunks)}")

# 3. Векторизуємо та зберігаємо у FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Створення векторів... (це займе кілька секунд)")
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

faiss.write_index(index, "refrigerator_kb.index")
print("Векторну базу знань успішно створено та збережено!")