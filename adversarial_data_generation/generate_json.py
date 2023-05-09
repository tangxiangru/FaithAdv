import pandas as pd
import numpy as np
import json
import random
from datasets import load_dataset
from nltk import sent_tokenize, word_tokenize

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    # return sent
    return sent.replace("\n", " ")

if __name__ == "__main__":
    num_to_generate = 17
    filePath_to_save = "adversarial_data_generation/json_file/sents5.json"

    cnndm = {
        "datasetName": "cnn_dailymail",
        "version": "3.0.0",
        "split": "train"
    }
    dataset = load_dataset(cnndm["datasetName"], cnndm["version"], split=cnndm["split"])
    examples = []

    indexes = random.sample(range(0, dataset.num_rows), num_to_generate)
    for i in indexes:
        item = dataset[i]
        summary = item["highlights"]
        summary_sents = sent_tokenize(summary)
        index = random.randint(0, len(summary_sents)-1)
        sentence = remove_return(summary_sents[index])

        temp = {
            "id": item["id"],
            "summary": summary,
            "sentence": sentence,
            "perturbed_sentence": ""
        }
        examples.append(temp)

    with open(filePath_to_save, "w") as f:
        json.dump(examples, f, indent=4)
