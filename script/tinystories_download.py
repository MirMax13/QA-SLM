from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

ds["train"].to_json("tinystories_train.json", orient="records", lines=True)
ds["validation"].to_json("tinystories_val.json", orient="records", lines=True)
