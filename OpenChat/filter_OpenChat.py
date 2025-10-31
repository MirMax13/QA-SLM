import json
from time import sleep
from dotenv import load_dotenv
from config.config import OUTPUT_JSON_CLEANED

load_dotenv()

# shared helpers
from OpenChat.common import filter_qa_candidates


# ========== STEP 6: Main loop ==========
def main():
    print(OUTPUT_JSON_CLEANED)
    # read qas from json
    mode ='piqa'
    dataset_cleaned = []
    with open("fridge_dataset_piqa.json", "r", encoding="utf-8") as f:
        all_qas = json.load(f)
    

    filtered_qas = filter_qa_candidates(all_qas)
    dataset_cleaned.extend(filtered_qas)
    print(f"✅ {len(filtered_qas)} kept after filtering")
    sleep(2)
    

    with open(OUTPUT_JSON_CLEANED, "w", encoding="utf-8") as f:
        json.dump(dataset_cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(dataset_cleaned)} cleaned entries (after filtering)")
if __name__ == "__main__":
    main()
