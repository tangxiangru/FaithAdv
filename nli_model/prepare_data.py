from datasets import load_dataset
import random
from utils.dataset_utils import *
from nltk import sent_tokenize
from tqdm import tqdm
import json
import random
import os

random.seed(42)

def prepare_pos_example(output_path):
    outputs = []

    sampled_ids = random.sample(range(len(dataset)), 50000)
    for i in tqdm(sampled_ids):
        example = dataset[i]
        data_id = example['id']
        source_doc = example['article']
        summary = example['highlights']

        summary_sents = sent_tokenize(summary)
        selected_summary_sent = random.sample(summary_sents, 1)[0]
        
        source_sents = sent_tokenize(source_doc)
        
        topn_source_sents, topn_scores, topn_source_sent_ids = get_topn_sents(source_sents, selected_summary_sent, topn, score_fn)

        output_example = {
            "id": f"{data_id}-1",
            "topn_source_sents": topn_source_sents,
            "topn_scores": topn_scores,
            "summary_sent": selected_summary_sent,
            "topn_source_sent_ids": topn_source_sent_ids,
            "label": 1
        }
        
        outputs.append(output_example)

    json.dump(outputs, open(output_path, "w"), indent=4)
    print(f"Saved {len(outputs)} examples to {output_path}")

def prepare_neg_example(input_path, output_path):
    id_source_doc_dict = {}
    outputs = []

    for i in range(len(dataset)):
        example = dataset[i]
        data_id = example['id']
        source_doc = example['article']
        id_source_doc_dict[data_id] = source_doc
    
    raw_pos_examples = json.load(open(input_path, "r"))
    for example in tqdm(raw_pos_examples):
        data_id = example["id"]
        source_doc = id_source_doc_dict[data_id]
        source_sents = sent_tokenize(source_doc)

        selected_summary_sent = example["sentence"]

        topn_source_sents, topn_scores, topn_source_sent_ids = get_topn_sents(source_sents, selected_summary_sent, topn, score_fn)

        output_example = {
            "id": f"{data_id}-0",
            "topn_source_sents": topn_source_sents,
            "topn_scores": topn_scores,
            "summary_sent": example["perturbed_sentence"],
            "topn_source_sent_ids": topn_source_sent_ids,
            "label": 0
        }

        outputs.append(output_example)

    json.dump(outputs, open(output_path, "w"), indent=4)
    print(f"Saved {len(outputs)} examples to {output_path}")

def prepare_training_example(neg_output_path, pos_output_path, final_output_dir, sample_num):
    neg_examples = json.load(open(neg_output_path, "r"))
    pos_examples = json.load(open(pos_output_path, "r"))

    neg_examples = random.sample(neg_examples, sample_num)
    pos_examples = random.sample(pos_examples, sample_num)

    final_examples = neg_examples + pos_examples
    random.shuffle(final_examples)

    # split into train, dev, test
    train_examples = final_examples[:int(len(final_examples) * 0.8)]
    dev_examples = final_examples[int(len(final_examples) * 0.8):int(len(final_examples) * 0.9)]
    test_examples = final_examples[int(len(final_examples) * 0.9):]

    # store in jsonl
    os.makedirs(final_output_dir, exist_ok=True)
    for split, examples in zip(["train", "dev", "test"], [train_examples, dev_examples, test_examples]):
        output_examples = []
        
        for example in tqdm(examples):
            output_example = {
                "id": example["id"],
                "premise": " ".join(example["topn_source_sents"]),
                "hypothesis": example["summary_sent"],
                "label": "ENTAILMENT" if example["label"] == 1 else "CONTRADICTION",
            }
            output_examples.append(output_example)
        json.dump(output_examples, open(f"{final_output_dir}/{split}.json", "w"), indent=4)
        print(f"Saved {len(examples)} examples to {final_output_dir}/{split}.json")


if __name__ == "__main__":
    score_fn = evaluate.load("bertscore")
    topn = 5
    
    
    neg_raw_data_path = "../adversarial_data_generation/json_file/sentences_processed.json"
    neg_output_path = "data/neg_examples.json"
    pos_output_path = "data/pos_examples.json"

    # dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    # prepare_neg_example(neg_raw_data_path, neg_output_path)
    # prepare_pos_example(pos_output_path)

    prepare_training_example(neg_output_path, pos_output_path, "data/nli_training_data", 10000)


    
