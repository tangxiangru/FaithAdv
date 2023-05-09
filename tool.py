from main import *
from tqdm import tqdm
import re
import time
from interruptingcow import timeout
from typing import List, Tuple
import copy
from tqdm import tqdm

def check_item_in_list_exist(sent: str, items: List[str]) -> bool:
    for item in items:
        if item in sent:
            return True
    return False

def generate_sentences(num: int, datasetName = "cnn_dailymail", version = "3.0.0", split = "train") -> List[str]:
    """
    Generate `num` sentences from the dataset.
    """
    dataset = load_dataset(datasetName, version, split=split)
    picked_sents = []
    count = num
    nn = dataset.num_rows
    nn = int(nn)
    print_hint_with_color("nn = " + str(nn), "cyan", nn.__class__)
    cur_id = 0
    with tqdm(total=int(nn)) as pbar:
        while cur_id < nn:
            if count == 0:
                break
            item = dataset[cur_id]["highlights"]
            claim_sents = sent_tokenize(item)
            for j in range(len(claim_sents)):
                sent = remove_return(claim_sents[j])

                # number exists in the sentence
                if not check_item_in_list_exist(sent, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                    continue
                # month exists in the sentence
                if not check_item_in_list_exist(sent, ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]):
                    continue
                # month exists in the sentence
                if not check_item_in_list_exist(sent, [" in ", " on ", " at "]):
                    continue
                # sequence exists in the sentence
                if not check_item_in_list_exist(sent, ["after", "before"]):
                    continue

                # check if there are four continuous digits in the sentence
                year_exist = re.search(r"\d{4}", sent)
                if year_exist is None:
                    continue

                picked_sents.append(sent)
                count -= 1
                if count == 0:
                    break
            cur_id += 1
            pbar.update(1)
    picked_sents = list(set(picked_sents))
    return picked_sents

def sents(quantity: int) -> List[str]:
    # generate sentences
    sents = generate_sentences(quantity, datasetName = "cnn_dailymail", version = "3.0.0", split = "train")
    path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/temp_data/num_month_in_before.json"
    json_write(path, sents)
    return sents

def get_stat_of_prompt():
    path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/prompt/prompt.json"
    prompt = json_read(path)
    cases = prompt["msg"]["cases"]
    assis = cases["assistant"]

    count = 0
    for sent in sents:
        if "after" in sent or "before" in sent:
            count += 1
    print(count)

def find_longest_same_substring(str1: str, str2: str) -> list:
    """
    find the longest same substring in two sentences (strings).
    return a list of the start position in each sentence and the length of the substring.
    """
    # str1 = "999aqrq bb cc"
    # str2 = "aqrq bb qq"
    len1 = len(str1)
    len2 = len(str2)
    res = []
    for i in range(len1):
        for j in range(len2):
            if str1[i] == str2[j]:
                # find the same character
                # check the next character
                k = 1
                while i + k < len1 and j + k < len2:
                    if str1[i + k] == str2[j + k]:
                        k += 1
                    else:
                        break
                res.append((i, k))
    return res

def replace_substr(string: str, start: int, length: int, new_substring: str) -> str:
    """
    replace the substring in the string with the new substring.
    """
    # string = "999aqrq bb cc"
    # start = 3
    # length = 5
    # new_substring = "qq"
    res = string[:start] + new_substring + string[start + length:]
    return res

def replace_replica(sent1: str, sent2: str, replace: str = "`", replica_flag = " ... ") -> Tuple[str, str]:
    """
    replace the replica in the two sentences with replace.
    """
    # sent1 = "The package was found in the building's mail room ."
    # sent2 = "The package was found in the building's mail room in Paris."
    res = find_longest(sent1, sent2)
    while True:
        if res is None:
            break
        if res[1] < 3:
            break
        substring = sent1[res[0] : res[0] + res[1]]
        sent1 = sent1.replace(substring, replace, 1)
        sent2 = sent2.replace(substring, replace, 1)
        res = find_longest(sent1, sent2)
    
    sent1 = sent1.replace(replace, replica_flag)
    sent2 = sent2.replace(replace, replica_flag)
    return sent1, sent2

def find_longest(sent1: str, sent2: str) -> Tuple[str, int]:
    res = find_longest_same_substring(sent1, sent2)
    if res is None:
        return None
    longest = res[0]
    for item in res:
        if item[1] > longest[1]:
            longest = item
    return longest

def json_convert(res_dir: str):
    files = os.listdir(res_dir)
    for file in files:
        if file[-9:] == "_new.json" or file == "backup":
            continue
        path = os.path.join(res_dir, file)
        print_hint_with_color("processing file: ", "green", path)
        sents = json_read(path)
        sent_pairs = []
        for sent in sents:
            res = sent["result"]
            for type_ in res:
                type_sent = res[type_]
                if type_sent["pert_sent"] != "no_sent":
                    org_, new_ = replace_replica(type_sent["orgn_sent"], type_sent["pert_sent"])
                    item = copy.deepcopy(type_sent)
                    item["orgn_"] = org_
                    item["pert_"] = new_
                else:
                    item = copy.deepcopy(type_sent)
                    item["orgn_"] = "no_sent"
                    item["pert_"] = "no_sent"

                sent_pairs.append(item)
        new_path = os.path.join(res_dir, file[:-5] + "_new.json")
        json_write(new_path, sent_pairs)
    return

def vision4check(file_dir: str):
    res_dir = os.getcwd() + "/output/" + file_dir
    json_convert(res_dir)
    print_hint_with_color("json_convert finished", "green")
  
def temp():
    path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/output/positive_1/positive.json"
    res = json_read(path)
    for item in res:
        r = item["result"]
        for i in r:
            if i in ["circumstance", "complex"]:
                r[i]["pert_sent"] = r[i]["response"]
    json_write(path, res)

def split_items(json_path: str, ratio: list):
    """
    split the sents into train, dev, test.
    """
    sents = json_read(json_path)
    length = len(sents)
    cp = length * ratio[0] // (ratio[0] + ratio[1])
    sent_0 = sents[:cp]
    sent_1 = sents[cp:]
    sent_0_path = json_path[:-5] + "_0.json"
    sent_1_path = json_path[:-5] + "_1.json"
    json_write(sent_0_path, sent_0)
    json_write(sent_1_path, sent_1)
    return

def temppp(path: str):
    sents = json_read(path)
    num = 0
    for sent in sents:
        sent["status"] = "not_started"
        sent["result"] = {}
        claim = sent["claim"]
        num += 1
        for err_type in ["person", "predicate", "entity", "subject or object of a predicate", "circumstance", "complex"]:
            tt = {
                "finished": False, 
                "err_type": err_type,
                "orgn_sent": claim,
                "pert_sent": "",
                "response": "",
                "elapsed_time": -1
            }
            sent["result"][err_type] = tt
    path = path[:-5] + "_new.json"
    json_write(path, sents)
    print(num)
    print_hint_with_color("finished", "green")
    return

def get_finished(path: str):
    sents = json_read(path)
    finished = []
    for sent in sents:
        if sent["status"] == "finished":
            finished.append(sent)
    path_finished = path[:-5] + "_finished.json"
    json_write(path_finished, finished)
    print_hint_with_color("finished", "green")
    return path_finished

def generate_id(index: int, length: int) -> str:
    if index < 0:
        raise ValueError("index should be positive")
    if length < 0:
        raise ValueError("length should be positive")
    str_num = str(index)
    index_str = str_num.zfill(length)
    return index_str

def add_id(path: str):
    sents = json_read(path)
    length = len(sents)
    for i in range(length):
        sents[i]["id"] = "pos_" + generate_id(i, 8)
        for type_ in sents[i]["result"]:
            type_name = type_.replace(" ", "")
            type_id = type_name.zfill(32)
            sents[i]["result"][type_]["id"] = sents[i]["id"] + "_" + type_id
    path = path[:-5] + "_id.json"
    json_write(path, sents)
    return

def split_train_dev_test(path: str, ratio: list):
    sents = json_read(path)
    random.shuffle(sents)
    length = len(sents)
    cp_0 = length * ratio[0] // (ratio[0] + ratio[1] + ratio[2])
    cp_1 = length * (ratio[0] + ratio[1]) // (ratio[0] + ratio[1] + ratio[2])
    sent_0 = sents[:cp_0]
    sent_1 = sents[cp_0:cp_1]
    sent_2 = sents[cp_1:]
    sent_0_path = path[:-5] + "_train.json"
    sent_1_path = path[:-5] + "_dev.json"
    sent_2_path = path[:-5] + "_test.json"
    json_write(sent_0_path, sent_0)
    json_write(sent_1_path, sent_1)
    json_write(sent_2_path, sent_2)
    path = path[:-5] + "_info.json"
    info = {
        "train": {
            "length": len(sent_0),
            "ids": [item["id"] for item in sent_0]
        },
        "dev": {
            "length": len(sent_1),
            "ids": [item["id"] for item in sent_1]
        },
        "test": {
            "length": len(sent_2),
            "ids": [item["id"] for item in sent_2]
        }
    }
    split_info_path = path[:-5] + "_split_info.json"
    json_write(split_info_path, info)
    print_hint_with_color("split finished", "green")
    return

def get_id_sent(path: str):
    sents = json_read(path)
    sent_list = []
    for sent in sents:
        temp = {
            "id": sent["id"],
            "text": sent["text"],
            "claim": sent["claim"]
        }
        sent_list.append(temp)
    path = path[:-5] + "_id_sent.json"
    json_write(path, sent_list)
    return

def generate_file_for_training(path: str, label: bool):
    sents = json_read(path)
    res = []
    for sent in sents:
        text = sent["text"]
        claim = sent["claim"]
        for type_ in sent["result"]:
            id = sent["result"][type_]["id"]
            pert_sent = sent["result"][type_]["pert_sent"]
            if pert_sent == "no_sent":
                continue
            temp = {
                "document_id": id,
                "original_document": text,
                "original_summary": claim,
                "perturbed_summary": pert_sent,
                "label": label
            }
            res.append(temp)
    path = path[:-5] + "_for_training.json"
    json_write(path, res)
    print_hint_with_color("generate file for training finished: ", "green", path)
    return

def claim_to_id(id_table_path: str, claim: str) -> str:
    id_table = json_read(id_table_path)
    for item in id_table:
        if item["claim"] == claim:
            return item["id"]
    return ""
    

def add_id_to_file(id_table_path: str, file_path: str):
    file = json_read(file_path)
    pbar = tqdm(file)
    pbar.set_description("Add id")
    for item in pbar:
        claim = item["claim"]
        id = claim_to_id(id_table_path, claim)
        if id == "":
            print_hint_with_color("id not found: ", "red", claim[:20])
        item["id"] = id
        for key in item["result"]:
            item["result"][key]["id"] = id + "_" + key.zfill(32)
    path = file_path[:-5] + "_add_id.json"
    json_write(path, file)
    print_hint_with_color("add id finished", "green")
    return

def test_api(api: str):
    messages=[
        {"role": "system", "content": "You are MaBaoguo, who is pretty famous in China."},
        {"role": "user", "content": "When did you beat the British Hercules?"},
    ]
    openai.api_key = api
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messages,
        temperature=0.2, # 0.7
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    res = response['choices'][0]['message']['content']
    print_hint_with_color("res: ", "green", res)
    return

def run_this_id(p_id: int, num: int, i: int) -> bool:
    seg = 100
    seg_id = (i // seg) % num
    if i % num == (p_id + seg_id) % num:
        return True
    else:
        return False

def check_id_lost(path):
    sents = json_read(path)
    ids = [item["id"] for item in sents]
    ids_int = [int(id[-8:]) for id in ids]
    ids_int.sort()
    len_of_ids = len(ids_int)
    print_hint_with_color("len of ids: ", "green", len_of_ids)
    num_lost = 0
    ids_check = []
    for i in range(len(ids_int)):
        c_ids_int = ids_int[i]
        if c_ids_int in ids_check:
            print_hint_with_color("id repeat: ", "red", c_ids_int)
        ids_check.append(c_ids_int)
    supposed = range(len(ids_int))
    got = list(set(ids_check))
    lost = list(set(supposed) - set(got))
    print_hint_with_color("lost: ", "red", lost)
    return

def get_id(sent:dict) -> int:
    id = sent["id"]
    return int(id[-8:])

def sort_by_id(path):
    sents = json_read(path)
    sents.sort(key=get_id)
    path = path[:-5] + "_sorted.json"
    json_write(path, sents)
    print_hint_with_color("sort finished: ", "green", path)
    return

def find_set_by_id(id: str) -> str:
    split_info_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/split_info.json"
    split_info = json_read(split_info_path)
    if id in split_info["train"]["ids"]:
        return "train"
    elif id in split_info["dev"]["ids"]:
        return "dev"
    elif id in split_info["test"]["ids"]:
        return "test"
    else:
        print_hint_with_color("id not found: ", "red", id)
        return ""

def change_id(path: str):
    sents = json_read(path)
    for sent in sents:
        id = sent["result"]["subject or object of a predicate"]["id"]
        sent["result"]["subject or object of a predicate"]["id"] = id.replace(" ", "")
    path = path[:-5] + "_changed_id.json"
    json_write(path, sents)
    print_hint_with_color("change id finished: ", "green", path)
    return

def split_file_by_id(path: str):
    sents = json_read(path)
    train = []
    dev = []
    test = []
    err = []
    pbar = tqdm(sents)
    pbar.set_description("Split file")
    for sent in pbar:
        id = sent["id"]
        text = sent["text"]
        set = find_set_by_id(id)
        if set == "train":
            for item in sent["result"]:
                info = sent["result"][item]
                if info["pert_sent"] == "no_sent":
                    continue
                else:
                    temp = {
                        "document_id": info["id"],
                        "original_document": text,
                        "original_summary": info["orgn_sent"],
                        "perturbed_summary": info["pert_sent"],
                        "label": True
                    }
                    train.append(temp)
        elif set == "dev":
            for item in sent["result"]:
                info = sent["result"][item]
                if info["pert_sent"] == "no_sent":
                    continue
                else:
                    temp = {
                        "document_id": info["id"],
                        "original_document": text,
                        "original_summary": info["orgn_sent"],
                        "perturbed_summary": info["pert_sent"],
                        "label": True
                    }
                    dev.append(temp)
        elif set == "test":
            for item in sent["result"]:
                info = sent["result"][item]
                if info["pert_sent"] == "no_sent":
                    continue
                else:
                    temp = {
                        "document_id": info["id"],
                        "original_document": text,
                        "original_summary": info["orgn_sent"],
                        "perturbed_summary": info["pert_sent"],
                        "label": True
                    }
                    test.append(temp)
        else:
            err.append(id)
            print_hint_with_color("set not found: ", "red", id)
    temp_path = path[:-10] + "_train.json"
    json_write(temp_path, train)
    temp_path = path[:-10] + "_dev.json"
    json_write(temp_path, dev)
    temp_path = path[:-10] + "_test.json"
    json_write(temp_path, test)
    if len(err) != 0:
        temp_path = path[:-10] + "_err.json"
        json_write(temp_path, err)
        print_hint_with_color("err: ", "red", err)
    else:
        print_hint_with_color("no err: ", "green", err)

    print_hint_with_color("split finished: ", "green", path)
    return

def modify_id(path: str):
    sents = json_read(path)
    for sent in sents:
        id = sent["document_id"]
        sent["document_id"] = "neg" + id[3:]
    path = path[:-5] + "_mid.json"
    json_write(path, sents)
    print_hint_with_color("modify id finished: ", "green", path)
    return

def add_type(path: str):
    sents = json_read(path)
    for sent in sents:
        id = sent["document_id"]
        types = [
            "person", "predicate", "entity", "subjectorobjectofapredicate", "circumstance", "complex",
            "location", "time", "person", "number", "predicate", "subjectorobjectofapredicate", "pronoun", "negation"
            ]
        types = list(set(types))
        not_found = True
        for type in types:
            if type in id:
                not_found = False
                sent["type"] = type
                break
        if not_found:
            print_hint_with_color("type not found: ", "red", id)
    path = path[:-5] + "_mid.json"
    json_write(path, sents)
    print_hint_with_color("add type finished: ", "green", path)
    return

if __name__ == "__main__":
    # # path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/id_sent.json"
    # path = []
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/negative_final/neg_dev.json")
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/negative_final/neg_test.json")
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/negative_final/neg_train.json")
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/positive_final/pos_dev.json")
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/positive_final/pos_test.json")
    # path.append("/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_backup/positive_final/pos_train.json")
    # for item in path:
    #     add_type(item)

    pass