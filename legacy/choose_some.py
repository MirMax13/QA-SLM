import json
import random


input_file = "fridge_dataset_v3.1_clean.json"

output_file = "fridge_dataset_v3.1_clean_6000.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

sampled_data = random.sample(data, 6000)
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(sampled_data, f, ensure_ascii=False, indent=2)


#Split into train and test sets
train_size = 5000
train_data = sampled_data[:train_size]
test_data = sampled_data[train_size:]
with open("fridge_dataset_v3.1_clean_6000_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open("fridge_dataset_v3.1_clean_6000_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)