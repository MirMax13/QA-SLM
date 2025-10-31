import os
import json
import re
import requests
from time import sleep
from transformers import AutoTokenizer
try:
    # prefer package config
    from config.config import LM_API_URL, HEADERS, MODEL_NAME
except Exception:
    # fallback to environment variables if config import fails
    LM_API_URL = os.getenv("LM_API_URL")
    HEADERS = {"Content-Type": "application/json"}
    MODEL_NAME = os.getenv("MODEL_NAME")

_tokenizer = None

def num_tokens(text: str) -> int:
    """Return approximate token count for text using a tokenizer.

    The tokenizer is cached after first load to avoid repeated downloads.
    """
    global _tokenizer
    if _tokenizer is None:
        # try a known model id first, fall back to MODEL_NAME
        model_id = "openchat/openchat-3.6-8b-20240522"
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            if MODEL_NAME:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            else:
                # last resort: load a small generic tokenizer
                _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        # conservative fallback
        return max(1, len(text.split()) // 1)


def call_lm(messages, temperature=0.7, max_tokens=300, timeout=60, retries=3, backoff_factor=2.0):
    """Call an LM HTTP endpoint with retries and basic diagnostics.

    Returns a string (assistant content) or empty string on failure.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        concat = "\n".join([m.get("content", "") for m in messages])
        approx_tokens = num_tokens(concat)
    except Exception:
        approx_tokens = None

    if approx_tokens is not None:
        print(f"→ Approx tokens in prompt: {approx_tokens}")

    attempt = 0
    while attempt < retries:
        try:
            attempt += 1
            resp = requests.post(LM_API_URL, headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # compatibility: many shapes
            if isinstance(data, dict):
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    ch = data["choices"][0]
                    if isinstance(ch, dict) and ch.get("message") and ch["message"].get("content"):
                        return ch["message"]["content"]
                    if isinstance(ch, dict) and ch.get("text"):
                        return ch["text"]
                if "output_text" in data and data.get("output_text"):
                    return data.get("output_text")
                if data.get("content"):
                    return data.get("content")

            try:
                return str(data)
            except Exception:
                return ""

        except requests.exceptions.Timeout as e:
            print(f"⚠️ LM call timeout (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                wait = backoff_factor ** (attempt - 1)
                print(f"→ retrying in {wait:.1f}s...")
                sleep(wait)
                continue
            else:
                print("❌ LM call timed out after retries")
                return ""
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ Connection error to LM (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                wait = backoff_factor ** (attempt - 1)
                print(f"→ retrying in {wait:.1f}s...")
                sleep(wait)
                continue
            else:
                print("❌ Connection failed after retries")
                return ""
        except Exception as e:
            print(f"❌ Error during LM call (attempt {attempt}/{retries}): {e}")
            return ""


def load_blocks_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.split(r'🔹 Блок \d+ \(\d+ слів\):\n[-]+\n(.*?)\n[-]+\n', content, flags=re.DOTALL)
    # re.split() returns [prefix, block1, block2, ...]
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    print(f"📄 Завантажено {len(blocks)} блоків із {file_path}")
    return blocks


def parse_qa_pairs(text):
    qas = []
    qa_blocks = re.findall(r"Q\d*:\s*(.*?)\s*A:\s*(.*?)(?=Q\d*:|$)", text, re.DOTALL)
    for q, a in qa_blocks:
        question = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', q.strip().replace("\n", " ")).strip()
        answer = re.sub(r'^(Paraphrase\s*\d+:|^\d+\.\s*)', '', a.strip().replace("\n", " ")).strip()
        if question and answer:
            qas.append({
                "instruction": question,
                "response": answer,
                "tag": "good",
            })
    print(f"🔍 Found {len(qas)} QA pairs")
    return qas


def save_qa(entry, file_path: str = "orig.json"):
    """Append generated QA(s) to a JSON array file.

    This uses the same lightweight append style as earlier scripts: it creates the file
    with an opening `[` and appends `,`+JSON for subsequent entries. This matches existing
    repository usage and avoids loading very large files into memory.
    """
    # allow passing a list
    entries = entry if isinstance(entry, list) else [entry]

    if os.path.exists(file_path):
        with open(file_path, "a", encoding="utf-8") as file:
            for e in entries:
                file.write(",\n")
                file.write(json.dumps(e, ensure_ascii=False, indent=2))
    else:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("[\n")
            for i, e in enumerate(entries):
                if i:
                    file.write(",\n")
                file.write(json.dumps(e, ensure_ascii=False, indent=2))


def filter_qa_candidates(qas, batch_size=35, out_file_prefix="filtered"):
    """Filter a list of QA dicts by asking the LM to mark good indices.

    Returns the cleaned list. Saves filtered results in files named f"{out_file_prefix}.json" via save_qa.
    """
    if not qas:
        return []

    cleaned = []
    total = len(qas)
    batches = [qas[i:i + batch_size] for i in range(0, total, batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"🔍 Filtering batch {b_idx+1}/{len(batches)} with {len(batch)} pairs")
        text = "\n".join([f"{i+1}. Q: {qa['instruction']}\n   A: {qa['response']}" for i, qa in enumerate(batch)])
        prompt = f"""
        <|im_start|>system
        You are a QA data cleaner. Review the following question-answer pairs and keep all that are:
        - grammatically correct
        - meaningful and relevant to a refrigerator manual
        - not exact duplicates (slightly reworded paraphrases are OK)

        Return a list of indices (e.g., 1, 2, 3, 5, 6).
        <|im_end|>
        <|im_start|>user
        {text}
        <|im_end|>
        <|im_start|>assistant
        """
        if num_tokens(prompt) > 8000:
            print(f"⚠️ Filtering prompt too long in batch {b_idx+1}")
        result = call_lm([{"role": "user", "content": prompt}])
        indices = set(int(i.strip()) for i in re.findall(r"\d+", result))
        kept = [batch[i - 1] for i in indices if 0 < i <= len(batch)]
        if kept:
            save_qa(kept, file_path=f"{out_file_prefix}.json")
        cleaned.extend(kept)

    return cleaned
