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

# 1. Завантажуємо векторизатор та збережений FAISS індекс
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("refrigerator_kb.index")

# Твій список chunks має бути завантажений так само, як у попередньому скрипті
chunks = load_chunks_from_file("input/Instructions/Instruction_v1.4.txt")

def ask_database(user_query):
    # Перетворюємо питання у вектор
    query_vector = embedder.encode([user_query], convert_to_numpy=True)
    
    # Шукаємо 1 найближчий фрагмент тексту
    distances, indices = index.search(query_vector, k=1)
    best_chunk = chunks[indices[0][0]]
    
    return best_chunk

# Тестуємо
context = ask_database("What is the optimal temperature for the freezer?")
user_query = "What is the optimal temperature for the freezer?"

# Формуємо RAG-промпт
prompt = f"""Use the following context to answer the question.
Context: {context}

Question: {user_query}
Answer:"""

print(prompt)