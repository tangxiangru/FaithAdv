import json
from tqdm import tqdm
import random
from datasets import load_dataset
from nltk import sent_tokenize

def process_advfact_data(jsonl_path):
    # read jsonl
    data = open(jsonl_path, "r").readlines()
    outputs = []
    pos, neg = 0, 0
    for i in tqdm(data):
        example = json.loads(i)
        if "augmentation" in example and example["augmentation"]:
            if "noise" in example and example["noise"]:
                continue
            if "backtranslation" in example and example["backtranslation"]:
                continue
            
            if example["label"] == "CORRECT":
                pos += 1
            else:
                neg += 1
            outputs.append(example)
    
    return random.sample(outputs, sample_num)

def prepare_our_neg_example(input_path):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    id_source_doc_dict = {}
    outputs = []
    
    for i in range(len(dataset)):
        example = dataset[i]
        data_id = example['id']
        source_doc = example['article']
        id_source_doc_dict[data_id] = source_doc
    
    raw_pos_examples = json.load(open(input_path, "r"))
    for example in tqdm(raw_pos_examples):
        data_id = example["id"].split("-")[0]
        source_doc = id_source_doc_dict[data_id]

        output_example = {
            "text": source_doc,
            "claim": example["summary_sent"],
            "label": "INCORRECT"
        }

        outputs.append(output_example)
    
    return random.sample(outputs, sample_num)

def prepare_pos_example():
    outputs = []
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    sampled_ids = random.sample(range(len(dataset)), sample_num)
    for i in tqdm(sampled_ids):
        example = dataset[i]
        data_id = example['id']
        source_doc = example['article']
        summary = example['highlights']

        summary_sents = sent_tokenize(summary)
        selected_summary_sent = random.sample(summary_sents, 1)[0]
        

        output_example = {
            "text": source_doc,
            "claim": selected_summary_sent,
            "label": "CORRECT"
        }
        
        outputs.append(output_example)

    return outputs


if __name__ == "__main__":
    sample_num = 20000
    advfact_neg_data = process_advfact_data("raw_data/advfact.jsonl")
    our_neg_data = prepare_our_neg_example("raw_data/neg_examples.json")
    pos_data = prepare_pos_example()
    
    advfact_data = advfact_neg_data + pos_data
    our_data = our_neg_data + pos_data
    
    random.shuffle(advfact_data)
    random.shuffle(our_data)
    
    json.dump(advfact_data, open("output_data/advfact_data.json", "w"), indent = 4)
    json.dump(our_data, open("output_data/our_data.json", "w"), indent = 4)
    
    print("advfact data size:", len(advfact_data))
    print("our data size:", len(our_data))
    