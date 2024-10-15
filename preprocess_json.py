import json
import uuid

with open("MLCQCodeSmellSamples.json", "r") as f:
    original_json_data = json.load(f)


for i, entry in enumerate(original_json_data, start=1):
    entry["unique_id"] = f"id_{i}"

with open("updated_data.json", "w") as f:
    json.dump(original_json_data, f, indent=4)

print("Added 'unique_id' field to each entry.")
