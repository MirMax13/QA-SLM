from openai import OpenAI
import json
import os
import uuid
from datetime import datetime
from config.config import MODEL_NAME, OPENAI_API_KEY, NOUN_PATH, VERB_PATH, ADJ_PATH
from .utils import safe_gpt_call
import random

client = OpenAI(api_key=OPENAI_API_KEY)
def read_word_from_txt(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        words = f.read().splitlines()
    return words

def generate_story(verb: str, noun: str, adj: str, feature: str, ending: str, temperature: float = 0.8):
    """Generate a short fridge-related story (3-5 paragraphs) using provided lexical constraints."""
    prompt = (
        f"Write a short story (3-5 paragraphs) about a person and their smart fridge. "
        f"The story must use the verb '{verb}', noun '{noun}', and adjective '{adj}'. "
        f"The story must naturally contain {feature} and end with a {ending}, "
        "but do not mention these words directly."
        "Use only simple words that could appear in a refrigerator instruction manual. "
        "Avoid complex literary wording. Keep it clear, concrete, and practical. "
        "Do not use labels like 'The moral:' or 'The foreshadowing:'. "
        "Return ONLY the story text (no explanations)."
    )

    system_msg = (
        "You are a helpful writing assistant that produces simple, clear training stories about a smart fridge. "
        "Respect the lexical constraints. Maintain 3-5 short paragraphs separated by blank lines. "
        "Do not use meta commentary or explicit labels; let features be shown through the story itself."
    )

    def _call():
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            reasoning={
                "effort": "minimal",
                "summary": None,
            },
            max_output_tokens=1100,
            # temperature=temperature,
        )
        story = response.output_text.strip()
        usage = response.usage.to_dict()
        return story, usage
    story, usage = safe_gpt_call(_call)

    return story, usage

def save_story(entry: dict, file_path: str = "stories.json"):
    """Append a generated story record to a JSON file (list of entries)."""
    if os.path.exists(file_path):
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(",\n")
            file.write(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("[\n")
            file.write(json.dumps(entry, ensure_ascii=False, indent=2))
            # file.write("\n]")

    print(f"💾 Saved story {entry['id']}")


def main():
    nouns = read_word_from_txt(NOUN_PATH)
    verbs = read_word_from_txt(VERB_PATH)
    adjectives = read_word_from_txt(ADJ_PATH)
    features = ["dialogue", "moral value", "plot twist", "foreshadowing", "conflict"]
    endings = ["bad ending", "good ending", "relevant", "irrelevant"]  # weights correspond below

    print("🚀 Story generation mode. Press Ctrl+C to stop at any time.")
    story_count = 0
    try:
        while True:
            random_noun = random.choice(nouns)
            random_verb = random.choice(verbs)
            random_adjective = random.choice(adjectives)
            random_feature = random.choice(features)
            random_ending = random.choices(endings, weights=[33, 33, 30, 4], k=1)[0]

            print("\n🔤 Selected tokens:")
            print(f"  Noun: {random_noun}")
            print(f"  Verb: {random_verb}")
            print(f"  Adjective: {random_adjective}")
            print(f"  Feature: {random_feature}")
            print(f"  Ending: {random_ending}")

            story, usage = generate_story(random_verb, random_noun, random_adjective, random_feature, random_ending)
            if story is None:
                print("❌ Story generation failed, skipping.")
                continue

            print("\n📘 Story:\n" + story + "\n")

            entry = {
                "id": str(uuid.uuid4()),
                "story": story,
                "metadata": {
                    "verb": random_verb,
                    "noun": random_noun,
                    "adjective": random_adjective,
                    "feature": random_feature,
                    "ending": random_ending,
                    "model": MODEL_NAME,
                    "timestamp": datetime.utcnow().isoformat()+"Z",
                    "usage": usage
                }
            }
            save_story(entry)
            story_count += 1
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    print(f"✅ Finished. Total stories generated: {story_count}")


if __name__ == "__main__":
    main()
