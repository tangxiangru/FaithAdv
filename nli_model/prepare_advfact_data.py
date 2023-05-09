from utils.dataset_utils import *
from nltk import sent_tokenize
import os
import json
from tqdm import tqdm
import random

def get_advfact_paths(input_dir):
    paths = []
    test_set_dirs = [i for i in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, i))]
    for test_set_dir in test_set_dirs:
        subset_dirs = [i for i in os.listdir(os.path.join(input_dir, test_set_dir)) if os.path.isdir(os.path.join(input_dir, test_set_dir, i))]

        paths.extend([os.path.join(input_dir, test_set_dir, subset_dir, "data-dev.jsonl") for subset_dir in subset_dirs])
    
    return paths

def process_single_advfact_data(file_path, topn=5, truncate = True):
    # read jsonl file
    test_set, subset = file_path.split("/")[-3:-1]
    data = open(file_path, "r").readlines()

    if truncate:
        truncate_num = 500
        if len(data) > truncate_num:
            data = random.sample(data, truncate_num)

    outputs = []
    for i in tqdm(data):
        example = json.loads(i)
        data_id = f"{example['id']}-{test_set}-{subset}"
        label = "CONTRADICTION" if example["label"] == "INCORRECT" else "ENTAILMENT"

        source_doc = example["text"]
        source_sents = sent_tokenize(source_doc)

        selected_summary_sent = example["claim"]

        topn_source_sents, topn_scores, topn_source_sent_ids = get_topn_sents(source_sents, selected_summary_sent, topn, score_fn)

        output_example = {
            "id": data_id,
            "premise": " ".join(topn_source_sents),
            "hypothesis": selected_summary_sent,
            "label": label
        }

        outputs.append(output_example)
    return outputs

def prepare_advfact_data(file_paths, topn=5, trauncate=True):
    outputs = []
    for file_path in tqdm(file_paths):
        outputs.extend(process_single_advfact_data(file_path, topn, trauncate))
    
    json.dump(outputs, open(f"advfact_data.json", "w"), indent=4)

    return outputs

if __name__ == "__main__":
    score_fn = evaluate.load("bertscore")
    input_dir = "advfact_data"
    file_paths = get_advfact_paths(input_dir)
    outputs = prepare_advfact_data(file_paths, topn=5, trauncate=False)




