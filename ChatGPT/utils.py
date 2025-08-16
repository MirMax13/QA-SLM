import openai
from time import sleep
import json
import os
import re
import random
from config.config import MODEL_NAME, MODEL_NAME_2, OPENAI_API_KEY, GENERATIVE
openai.api_key = OPENAI_API_KEY

def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except (openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.APIError) as e:
            print(f"⏳ Rate limit hit, sleeping 20s (attempt {attempt+1})")
            sleep(20)
    switch_model()  # Switch model if rate limit is hit
    # raise RuntimeError("Rate limit hit too many times.")
    try:
        return call_func(*args, **kwargs)
    except Exception as e:
        print(f"❌ GPT call failed after model switch: {e}")
        return None

def switch_model():
    global MODEL_NAME
    MODEL_NAME = MODEL_NAME_2
    print(f"🔄 Switched model to {MODEL_NAME}")

def call_lm(messages, model=MODEL_NAME, max_tokens=512, temperature=0.7):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens
    print(f"📊 Tokens used: prompt={prompt_tokens}, completion={completion_tokens}")

    token_stats = []
    # Save stats to global list
    if os.path.exists("token_stats.json"):
        with open("token_stats.json", "r") as f:
            token_stats = json.load(f)
    token_stats.append({
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "messages": messages[-1]["content"][:100]  # just preview
    })
    with open("token_stats.json", "w") as f:
        json.dump(token_stats, f, indent=2)

    return response["choices"][0]["message"]["content"]

def load_blocks_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and all("context" in block for block in data):
        print(f"📄 Завантажено {len(data)} блоків із {file_path}")
        return [block["context"] for block in data]
    else:
        raise ValueError(f"JSON file {file_path} does not contain valid blocks with 'context' keys")


def parse_qa_pairs(text, status="good"):
    qas = []
    
    # Спершу класичний формат Q:... A:...
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL)
    blocks = load_blocks_from_json("blocks.json")
    for q, a in qa_blocks: #TODO: maybe delete this loop
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', a.strip().replace("\n", " ")).strip()
        if question and answer:
            if GENERATIVE:
                if status == "irrelevant":
                    qas.append({"instruction": question, "response": answer, "tag": "irrelevant"})
                else:
                    qas.append({"instruction": question, "response": answer, "tag": "good"})
            else:
                context = random.choice(blocks)
                qas.append({"context": context, "question": question, "answers": [], "is_impossible": True})

    # Якщо нічого не знайшли — шукаємо формат "1. ... - ..."
    if not qas:
        alt_blocks = re.findall(r"\d+\.\s*(.*?)\n\s*[-•]\s*(.*?)(?=\n\d+\.|\Z)", text, re.DOTALL)
        for q, a in alt_blocks:
            question = q.strip().replace("\n", " ")
            answer = a.strip().replace("\n", " ")
            if question and answer:
                if GENERATIVE:
                    qas.append({"instruction": question, "response": answer, "tag": "irrelevant"})
                else:
                    context = random.choice(blocks)
                    qas.append({"context": context, "question": question, "answers": [], "is_impossible": True})
    print(f"🔍 Found {len(qas)} QA pairs")
    return qas