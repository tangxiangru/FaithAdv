import json
import os
import random

ours_path = "output_data/phase2/raw_ours_processed.json"
advfact_path = "output_data/phase2/advfact.json"

ours_outputs = json.load(open(ours_path, "r"))
advfact_outputs = json.load(open(advfact_path, "r"))

example_ids = set()

for example in ours_outputs:
    example_ids.add(example["id"])
    
for example in advfact_outputs:
    if example["label"] == "CORRECT":
        ours_outputs.append(example)
        
json.dump(ours_outputs, open("output_data/phase2/ours_data.json", "w"), indent=4)
        


