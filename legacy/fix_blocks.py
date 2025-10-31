import re

input_path = r"c:\All\Projects\QA-SLM\input\Instructions\Instruction_v1.4.txt"
output_path = r"c:\All\Projects\QA-SLM\input\Instructions\Instruction_v1.4.fixed.txt"

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Знайти всі блоки
block_pattern = re.compile(
    r"🔹 Блок (\d+) \((\d+) слів\):\n.*?\n(.*?)(?=\n🔹 Блок \d+ \(|\Z)", re.DOTALL
)
blocks = block_pattern.findall(text)
print(f"Знайдено {len(blocks)} блоків для перевірки.")
errors = []
fixed_blocks = []

for i, (block_num, word_count_str, block_text) in enumerate(blocks, start=1):
    block_num = int(block_num)
    word_count = int(word_count_str)
    # Перевірка нумерації
    if block_num != i:
        errors.append(f"❌ Block number mismatch: found {block_num}, expected {i}")
        block_num = i  # виправити
    # Порахувати слова
    # Враховуємо тільки текст блоку, без службових ліній
    block_main = re.sub(r"-{10,}.*", "", block_text, flags=re.DOTALL)
    words = re.findall(r"\w+", block_main)
    actual_count = len(words)
    if actual_count != word_count:
        errors.append(
            f"❌ Block {block_num}: word count mismatch (declared {word_count}, actual {actual_count})"
        )
        word_count = actual_count  # виправити
    # Зібрати виправлений блок
    fixed_blocks.append(
        f"🔹 Блок {block_num} ({word_count} слів):\n{'-'*60}\n{block_text.strip()}\n"
    )

# Записати виправлений файл
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(fixed_blocks))

print("Перевірка завершена.")
if errors:
    print("Знайдено помилки:")
    for err in errors:
        print(err)
    print(f"Виправлений файл записано у: {output_path}")
else:
    print("Всі блоки коректні. Виправлений файл також записано.")