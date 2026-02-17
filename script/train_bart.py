import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

# === КОНФІГУРАЦІЯ ===
MODEL_NAME = "facebook/bart-base"
DATA_FILE = "Full.jsonl"
OUTPUT_DIR = "./bart_finetuned_model"
WANDB_PROJECT = "bart-finetune-meluxina"
TARGET_LOSS = 0.20

# === 1. CALLBACK ДЛЯ ЗУПИНКИ ПРИ LOSS <= 0.20 ===
class StopOnLowLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Перевіряємо, чи є 'loss' (це training loss) у логах
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            if current_loss <= TARGET_LOSS:
                print(f"\n\n🛑 STOPPING CRITERIA MET: Training Loss {current_loss:.4f} <= {TARGET_LOSS}")
                control.should_training_stop = True

# === 2. ЗАВАНТАЖЕННЯ ДАНИХ ===
print("📄 Loading data...")
data_rows = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            # Фільтруємо "сміття", якщо раптом залишилося
            if item.get("tag") == "irrelevant": 
                continue
                
            question = item.get("instruction", "").strip()
            answer = item.get("response", "").strip()
            
            if question and answer:
                data_rows.append({
                    "input": f"question: {question}", 
                    "output": answer
                })
        except json.JSONDecodeError:
            continue

# Створюємо Dataset і ділимо на Train (90%) / Test (10%)
full_dataset = Dataset.from_list(data_rows)
dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"✅ Data loaded. Train: {len(train_dataset)}, Test: {len(eval_dataset)}")

# === 3. ТОКЕНІЗАЦІЯ ===
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = [doc for doc in examples["input"]]
    targets = [doc for doc in examples["output"]]
    
    # Токенізація входів
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    # Токенізація виходів (labels)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# === 4. МОДЕЛЬ ТА МЕТРИКИ ===
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# === 5. НАЛАШТУВАННЯ ТРЕНУВАННЯ ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",    # Оцінювати кожні N кроків
    eval_steps=100,                  # Частота перевірки
    logging_steps=10,               # Частота запису логів (важливо для нашого Callback!)
    save_steps=200,
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Можна 16, якщо A100
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=100,             # Ставимо із запасом, Callback зупинить раніше
    predict_with_generate=True,
    fp16=True,                      # Прискорення на GPU
    report_to="wandb",              # Вмикаємо WandB
    run_name="bart-fridge-run-01",
    load_best_model_at_end=True,    # Завантажити найкращу модель в кінці
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[StopOnLowLossCallback()] # <--- НАШ CALLBACK ТУТ
)

# === 6. ЗАПУСК ===
print("🚀 Starting training...")
trainer.train()

print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")