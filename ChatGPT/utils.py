import openai
from time import sleep
from config.config import MODEL_NAME, MODEL_NAME_2, OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def safe_gpt_call(call_func, *args, **kwargs):
    for attempt in range(5):
        try:
            return call_func(*args, **kwargs)
        except openai.error.RateLimitError:
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
