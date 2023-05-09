import os
import openai
import json
import argparse
from typing import List
from tqdm import tqdm
import time
import random
import pandas as pd
import numpy as np
import csv
from datasets import load_dataset
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
import multiprocessing as mp
from multiprocessing import Pool

from utils.read_api_key import read_api_keys
# random.seed(2023)

api_list = read_api_keys("adversarial_data_generation/utils/api_keys.txt", 100)

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    return sent.replace("\n", " ")

def generate_sentences(num: int, datasetName = "cnn_dailymail", version = "3.0.0", split = "train") -> List[str]:
    """
    Generate `num` sentences from the dataset.
    """
    dataset = load_dataset(datasetName, version, split=split)
    indexes = random.sample(range(0, dataset.num_rows), num)

    picked_sents = []
    for i in indexes:
        item = dataset[i]["highlights"]
        claim_sents = sent_tokenize(item)
        index = random.randint(0, len(claim_sents)-1)
        picked_sents.append(remove_return(claim_sents[index]))

    return picked_sents

def prepare_original_sents(quantity: int) -> List[str]:
    # quantity is the number of sentences to generate
    sents = generate_sentences(quantity, datasetName = "cnn_dailymail", version = "3.0.0", split = "train")
    return sents

def prepare_prompt(input_prompt_file, sep="#####") -> str:
    if input_prompt_file.endswith(".txt"):
        prompt_text = open(args.input_prompt).read()
    elif input_prompt_file.endswith(".csv"):
        df = pd.read_csv(input_prompt_file, sep=sep, engine='python', encoding='utf-8')
        original_sents = df["original_sent"].tolist()
        perturbed_sents = df["perturbed_sent"].tolist()
        prompt_text = "Change the meaning of the sentence by perturbing the sentence in word-level and ensure that the change is no more than 3 words.\n\n"
        for i in range(len(original_sents)):
            prompt_text += f'Original sentence: \n{original_sents[i]}\nPerturbed sentence: \n{perturbed_sents[i]}\n\n'
    return prompt_text.strip() + "\n\n"

def extract_perturbed_sent(response):
    response = response.strip()
    return response

def diff_is_one(words1: List[str], words2: List[str], same_meaning_words: List[str]):
    """
    sent1 is longer than sent2.
    the words in `same_meaning_words` should be lowercased.
    if diff is one of the words in `same_meaning_words`, then return True.
    otherwise, return False.
    """
    for item in words2:
        if item in words1:
            words1.remove(item)
        else:
            return False

    if words1[0].lower() in same_meaning_words:
        return True
    else:
        return False

def get_similarity(word1, word2):
    word1_set = wordnet.synsets(word1)
    word2_set = wordnet.synsets(word2)
    temp = [
        0 if word1.path_similarity(word2) == None 
        else word1.path_similarity(word2) 
        for word1 in word1_set for word2 in word2_set
        ]
    temp.append(0)
    return max(temp)

def get_synonyms(word: str):
    """
    get synonyms of a word.
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def same_len_with_synonyms(words1: List[str], words2: List[str], threshold = 0.5):
    """
    if there is only one word that is different between the two word lists, and the two words are synonyms, return True.
    otherwise, return False.
    """
    diff_words = []
    for i in range(len(words1)):
        if words1[i] != words2[i]:
            diff_words.append((words1[i], words2[i]))
    if len(diff_words) == 1:
        word1 = diff_words[0][0]
        word2 = diff_words[0][1]
        synonyms1 = get_synonyms(word1)
        synonyms2 = get_synonyms(word2)
        if word2 in synonyms1 or word1 in synonyms2:
            return True
        elif get_similarity(word1, word2) > threshold:
            return True
        else:
            return False
    else:
        return False

def gpt3_judge(sent1: str, sent2: str) -> str:
    """
    using GPT-3 to check if the two sentences have the same meaning.
    return the response from GPT-3.
    """
    # print(">>> func gpt3_judge called")
    # api = random.choice(api_list)
    prompt = f"Whether the two sentences maintain the same meaning? Output yes or no.\nsentence1: {sent1}\n sentence2: \n{sent2}\n"
    time.sleep(0.3)
    response = openai.Completion.create(
        # model=args.model,
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=16,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response = response["choices"][0]["text"]
    return response

def gpt3_same_meaning(sent1: str, sent2: str):
    """
    using GPT-3 to check.
    if the two sentences have the same meaning, return True.
    otherwise, return False.
    """
    response = gpt3_judge(sent1, sent2)
    if "yes" in str(response).lower():
        return "yes"
    elif "no" in str(response).lower():
        return "no"
    else:
        return "wrong"

def sents_is_same(sent: str, perturbed_sent: str, same_meaning_words: List[str], threshold: float):
    """
    the words in `same_meaning_words` should be lowercased.
    if the two sentences are the same, return True.
    if the two sentences are different by one word, return True.
    otherwise, return False.
    """
    words1 = word_tokenize(sent)
    words2 = word_tokenize(perturbed_sent)
    if words1 == words2:
        return True
    elif len(words1) == len(words2):
        slws = same_len_with_synonyms(words1, words2, threshold)
        if slws:
            return True
    elif abs(len(words1) - len(words2)) == 1:
        if len(words1) > len(words2):
            if diff_is_one(words1, words2, same_meaning_words):
                return True
        else:
            if diff_is_one(words2, words1, same_meaning_words):
                return True
    # else:
        # pass

    # maybe not good enough, but can be used to reject some sentences.
    gpt3_answer = gpt3_same_meaning(sent, perturbed_sent)
    if gpt3_answer == "yes":
        return True
    elif gpt3_answer == "no":
        return False
    else:
        print("gpt3 cannot jughe if the sentences have the same meaning.")
        return False

    # return False

def generate_perturbed_sent(sent: str, prompt_string_temp: str, threshold: float, debugging: bool = False):
    """
    generate perturbed sentence.
    garrantee that the perturbed sentence is different from the original sentence.
    """
    # used to test
    # sec = time.localtime(time.time()).tm_sec
    # if sec % 3 == 0:
    #     return "noSentGenerated", "noSentGenerated", "noSentGenerated"
    
    try_times = 3
    while try_times > 0:
        time.sleep(0.3)
        response = openai.Completion.create(
            # model=args.model,
            model="text-davinci-003",
            prompt=prompt_string_temp,
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        perturbed_sent = extract_perturbed_sent(response['choices'][0]["text"])
        same_meaning_words = []
        same_meaning_words.append("the")
        same_sents = sents_is_same(sent, perturbed_sent, same_meaning_words, threshold)
        if not same_sents:
            break
        else:
            try_times -= 1
            # print("same sentence. {} chances left to try.".format(try_times))
            if try_times == 0:
                break
    if try_times > 0:
        if debugging:
            judge = gpt3_judge(sent, perturbed_sent)
            judge = remove_return(judge)
            if len(judge) == 0:
                judge = "no result"
        else:
            judge = "no debugging, no judge"
        return sent, perturbed_sent, judge
    else:
        return "noSentGenerated", "noSentGenerated", "noSentGenerated"

def json_read(json_file_path: str):
    """
    read json file.
    """
    with open(json_file_path, "r") as f:
        json_input = json.load(f)
    return json_input

def json_write(json_file_path: str, json_input):
    """
    write json file.
    """
    with open(json_file_path, "w") as f:
        json.dump(json_input, f, indent=4)
    return

def json_check_file(file):
    """
    check if json file exists, if exists, return True, else create it and return False
    """
    if os.path.exists(file):
        return True
    else:
        with open(file, "w") as f:
            json.dump([], f)
        return False

def merge_json_files_to_one(file_path_list, file_path_output):
    """
    merge all the json files in file_path_list to one file.
    delete all the files in file_path_list.
    """
    sents = []
    for file_path in file_path_list:
        sents += json_read(file_path)
    json_write(file_path_output, sents)
    # delete all the files
    for file_path in file_path_list:
        os.remove(file_path)

def append_str_to_file_name(file:str, str_to_append:str) -> str:
    """
    append str_to_append to file name, before file extension
    """
    file_name, file_extension = os.path.splitext(file)
    return file_name + str_to_append + file_extension
    
def split_list_into_n_sublists(l:list, n:int) -> list:
    """
    split list l into n sublists
    """
    length = len(l)
    return [l[i*length // n: (i+1)*length // n] for i in range(n)]

def info_monitor(cur_id, deleted, file_path_info, interval: int = 0.1):
    json_info = json_read(file_path_info)
    while json_info["cur_id"] < json_info["sents_num"]:
        time.sleep(interval)
        changed = False
        json_info = json_read(file_path_info)
        if json_info["cur_id"] != cur_id.value:
            json_info["cur_id"] = cur_id.value
            changed = True
        if json_info["deleted"] != deleted.value:
            json_info["deleted"] = deleted.value
            changed = True
        if changed:
            json_write(file_path_info, json_info)

def process_sents_split(i, cur_id, deleted, api_set,
                        # para, 
                        csv_sep_read, word_similarity_threshold, debugging,
                        input_prompt, input_sentence,
                        ):
    prompt_string = prepare_prompt(input_prompt, sep=csv_sep_read)
    prompt_string_original = prepare_prompt(input_prompt, sep=csv_sep_read)
    # prompt_string = prepare_prompt(input_prompt, sep=para["csv_sep_read"])
    # prompt_string_original = prepare_prompt(input_prompt, sep=para["csv_sep_read"])
    file_input_sentence = append_str_to_file_name(input_sentence, f"_unprocessed_{i}")
    file_output_sentence = append_str_to_file_name(input_sentence, f"_processed_{i}")
    file_deleted_sentence = append_str_to_file_name(input_sentence, f"_deleted_{i}")

    sents = json_read(file_input_sentence)
    sents_processed = json_read(file_output_sentence)
    sents_deleted = json_read(file_deleted_sentence)

    cur_id_process = 0
    deleted_process = 0
    with tqdm(total=len(sents)) as pbar:
        while len(sents) > 0:
        # while cur_id_process - deleted_process < len(sents):
            openai.api_key = random.sample(list(api_set), 1)[0]
            try:
                # sent = sents[cur_id_process - deleted_process]["sentence"]
                sent = sents[0]["sentence"]
                sent = sent.strip()

                prompt_string += f'Original sentence: \n{sent}\nPerturbed sentence: \n'
                prompt_string_temp = prompt_string_original + f'Original sentence: \n{sent}\nPerturbed sentence: \n'

                # for i in range(para["replica_of_each_original_sent"]):
                sent, perturbed_sent, gpt3_result = generate_perturbed_sent(sent, prompt_string_temp, word_similarity_threshold, debugging)
                # sent, perturbed_sent, gpt3_result = generate_perturbed_sent(sent, prompt_string_temp, para["word_similarity_threshold"], para["debugging"])
                if sent == "noSentGenerated":
                    # print("no sent generated, delete this pair.")
                    # sents_deleted.append(sents[cur_id_process - deleted_process])
                    sents_deleted.append(sents[0])
                    # del sents[cur_id_process - deleted_process]
                    del sents[0]
                    deleted_process += 1
                    deleted.value += 1
                    json_write(file_input_sentence, sents)
                    json_write(file_deleted_sentence, sents_deleted)
                else:
                    # sents[cur_id_process - deleted_process]["perturbed_sentence"] = perturbed_sent
                    sents[0]["perturbed_sentence"] = perturbed_sent
                    # sents_processed.append(sents[cur_id_process - deleted_process])
                    sents_processed.append(sents[0])
                    # del sents[cur_id_process - deleted_process]
                    del sents[0]
                    json_write(file_input_sentence, sents)
                    json_write(file_output_sentence, sents_processed)

                cur_id_process += 1
                cur_id.value += 1
                pbar.update(1)

            except openai.error.RateLimitError as e:
                if e._message.startswith("You exceeded your current quota,"):
                    api_set.remove(openai.api_key)
                    print(f"Switching to API key: {openai.api_key}, {len(api_set)} API keys left.")
                else:
                    print(e._message[:50])
                time.sleep(20)
                error_flag = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="text-davinci-003", type=str)# text-curie-001
    parser.add_argument("--input_prompt", required=True, type=str)
    parser.add_argument("--input_sentence", default="./json_file/sents0.json", type=str)
    parser.add_argument("--core_number", default=1, type=int)
    parser.add_argument("--immediate_merge", default=False, type=bool)
    args = parser.parse_args()

    if args.core_number > 1:
        args.core_number -= 1

    file_path_info = append_str_to_file_name(args.input_sentence, "_info")
    
    if not args.immediate_merge:
        # >>> if exists, load info. if not, create info
        json_info_exist = json_check_file(file_path_info)
        sents = json_read(args.input_sentence)
        if json_info_exist:
            json_info = json_read(file_path_info)
        else:
            sents = json_read(args.input_sentence)
            json_info = {
                "started": False,
                "sents_num": len(sents),
                "cur_id": 0,
                "deleted": 0
            }
            json_write(file_path_info, json_info)

        # >>> according to json_info, if the process has not started, do some preparation
        json_temp_files = {
            "unprocessed": [],
            "processed": [],
            "deleted": []
        }
        if json_info["started"] == False:
            # sents = json_read(args.input_sentence)
            sents_split = split_list_into_n_sublists(sents, args.core_number)
            json_info["core_number"] = args.core_number
            for i in range(len(sents_split)):
                file_path = append_str_to_file_name(args.input_sentence, f"_unprocessed_{i}")
                json_temp_files["unprocessed"].append(file_path)
                json_write(file_path, sents_split[i])
                file_path = append_str_to_file_name(args.input_sentence, f"_processed_{i}")
                json_temp_files["processed"].append(file_path)
                json_write(file_path, [])
                file_path = append_str_to_file_name(args.input_sentence, f"_deleted_{i}")
                json_temp_files["deleted"].append(file_path)
                json_write(file_path, [])
            json_info["json_files"] = json_temp_files
            json_info["started"] = True
            json_write(file_path_info, json_info)
        file_number = json_info["core_number"]
        del sents

        # ########################## params to adjust manually ##########################
        para = {
            "quantity_of_original_sents_to_generate": 100,
            "replica_of_each_original_sent": 1,
            "output_dir": "outputs/result",
            "csv_sep_read": "ᚢ",
            "csv_sep_write": "ᚢ",     # "#####"
            "word_similarity_threshold": 0.6,
            "debugging": False,
            "json_process": True,
            "file_number": file_number
        }

        os.makedirs(para["output_dir"], exist_ok=True)

        api_set = set(api_list)
        # openai.api_key = random.sample(list(api_set), 1)[0]

        # cur_id and deleted are shared by all processes
        cur_id = mp.Value('i', json_info["cur_id"])
        deleted = mp.Value('i', json_info["deleted"])

        # >>> multiprocess, each process process one file with corresponding name
        json_process = []
        for i in range(file_number):
            json_process.append(mp.Process(
                    target=process_sents_split,
                    args=(i, cur_id, deleted, api_set,
                            para["csv_sep_read"], para["word_similarity_threshold"], para["debugging"], 
                            args.input_prompt, args.input_sentence, 
                            ),
                    kwargs=[]
                ))
        p_monitor = mp.Process(
                    target=info_monitor,
                    args=(cur_id, deleted, file_path_info, 0.01),
                    kwargs=[]
                )
        [p.start() for p in json_process]
        p_monitor.start()
        [p.join() for p in json_process]
        p_monitor.join()

    # >>> merge all files into one file
    
    json_info = json_read(file_path_info)
    json_temp_files = json_info["json_files"]
    merge_json_files_to_one(
        json_temp_files["unprocessed"],
        append_str_to_file_name(args.input_sentence, f"_unprocessed"))
    merge_json_files_to_one(
        json_temp_files["processed"],
        append_str_to_file_name(args.input_sentence, f"_processed"))
    merge_json_files_to_one(
        json_temp_files["deleted"],
        append_str_to_file_name(args.input_sentence, f"_deleted"))

    print(">>> All processes have been finished.")
    print(">>> suceeded: {}, failed: {}".format(json_info["sents_num"] - json_info["deleted"], json_info["deleted"]))
    print("    ############################ PROCESSES FINISHED ###########################")
    print()









# file
# - FILENAME_unprocessed_n
# - FILENAME_processed_n
# - FILENAME_deleted_n
# - FILENAME_info
#   - info {
        # started: False
        # sents_num: 52000
        # cur_id: 0
        # deleted: 0
        # json_files: {
        #   "unprocessed": [], 
        #   "processed": [], 
        #   "deleted": []
        # }
#       }