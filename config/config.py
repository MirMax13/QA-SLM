from dotenv import load_dotenv
import os

load_dotenv()

# Змінні конфігурації
PDF_PATH = os.getenv('PDF_PATH')
MODEL_NAME = os.getenv('MODEL_NAME_V2')
MODEL_NAME_2 = os.getenv('MODEL_NAME_V2.2')
OUTPUT_JSON = os.getenv('OUTPUT_JSON')
OUTPUT_JSON_CLEANED = os.getenv('OUTPUT_JSON_CLEANED')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USAGE_FILE = "api_usage.json"
ORIG_INPUT_JSON = os.getenv('ORIG_INPUT_JSON')
GENERATIVE = True
