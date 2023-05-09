import json
import os
import random

random.seed(42)

def process_one_file(jsonl_path, test_set_name, attack_name, max_example = 1000):
    data = open(jsonl_path, "r").readlines()
    outputs = []
    
    if len(data) > max_example:
        random.shuffle(data)
        data = data[:max_example]
        
    for i in data:
        example = json.loads(i)
        example["test_set_name"] = test_set_name
        example["attack_name"] = attack_name
        example["id"] = f"{example['id']}-{test_set_name}-{attack_name}"
        if "origin_claim" in example:
            outputs.append(example)
        
    return outputs


def prepare_advfact_data(input_dir):
    outputs = []
    perturb_count = 0
    
    subdirs = os.listdir(input_dir)
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        test_set_name = subdir
        attackdirs = os.listdir(subdir_path)
        for attackdir in attackdirs:
            attackdir_path = os.path.join(subdir_path, attackdir)
            attack_name = attackdir
            
            cur_data = process_one_file(os.path.join(attackdir_path, "data-dev.jsonl"), test_set_name, attack_name)
            outputs.extend(cur_data)
    
    for example in outputs:
        assert example["label"] in ["CORRECT", "INCORRECT"]
        if example["label"] == "INCORRECT":
            perturb_count += 1
            
    print(f"Total number of examples: {len(outputs)}")
    print(f"Total number of perturbed examples: {perturb_count}")
    return outputs
            
            
if __name__ == "__main__":
    input_dir = "raw_data/base_and_diagnostic_sets"
    output_path = "output_data/phase2/advfact.json"
    ours_outputs_path = "output_data/phase2/raw_ours.json"
    
    outputs = prepare_advfact_data(input_dir)
    json.dump(outputs, open(output_path, "w"), indent=4)
    
    ours_outputs = []
    
    for example in outputs:
        example["claim"] = ""
        if example["label"] == "INCORRECT":
            ours_outputs.append(example)
    json.dump(ours_outputs, open(ours_outputs_path, "w"), indent=4)        