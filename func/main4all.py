from datasets import load_dataset
from nltk import sent_tokenize
import pandas as pd
import numpy as np
import os
import json
import stat
import xlwt
import random
from tqdm import tqdm
from typing import Dict, Tuple, List
import sys
import traceback
import time

# loc_info_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/loc_info.json"
loc_info_path = "/root/autodl-tmp/loc_info.json"

### maybe only use once or twice #####
def delete_files_recursively(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path) and (
                file_path.endswith(".json") or file_path.endswith(".jsonl")
                ):
                os.remove(file_path)
    return

### reuse #####
def print_hint_with_color(hint: str, color: str, content: str = ""):
    """
    print hint with color.
    color: red, green, yellow, blue, purple, cyan, white, black
    """
    color_dict = {
        "red": "\033[1;31m",
        "green": "\033[1;32m",
        "yellow": "\033[1;33m",
        "blue": "\033[1;34m",
        "purple": "\033[1;35m",
        "cyan": "\033[1;36m",
        "white": "\033[1;37m",
        "black": "\033[1;30m",
        "end": "\033[0m",
    }
    print(color_dict[color] + hint + color_dict["end"] + content)
    return

def ask_if_continue():
    iinput = input("continue? (y/n): ")
    if iinput == "y":
        return
    else:
        exit(0)

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    return sent.replace("\n", " ")

def check_file_exists(file: str):
    """
    check if file exists, if exists, return True, else create it and return False
    """
    if not os.path.exists(file):
        # create the file
        with open(file, "w") as f:
            json.dump([], f)
        return False
    return True

def check_dir(dir):
    """
    check if dir exists, if exists, return True, else create it and return False
    """
    if os.path.exists(dir):
        return True
    else:
        os.makedirs(dir)
        return False

def json_read(json_file_path: str):
    """
    read json file.
    """
    with open(json_file_path, "r") as f:
        json_input = json.load(f)
    return json_input

def json_write(json_file_path: str, json_input):
    """
    write json into file.
    json_file_path: the path of the json file
    json_input: the input json
    """
    json_check_file(json_file_path)
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
        # get the dir of the file
        dir = os.path.dirname(file)
        check_dir(dir)
        with open(file, "w") as f:
            json.dump([], f)
        return False

def split_summary(dataset, output_file):
    datas = []
    article_id = -1
    rows = dataset.num_rows
    for i in range(rows):
        item = dataset[i]
        article_id += 1
        sents = sent_tokenize(item["abstract"])
        sent_id = -1
        for sent in sents:        
            sent_id += 1
            id = article_id * 1000 + sent_id
            temp = {
                "id": id,
                "article": item["article"],
                "claim": sent,
            }
            datas.append(temp)
    json_check_file(output_file)
    json_write(output_file, datas)

def split_summary_main():
    dataset_name = ["ccdv/pubmed-summarization", "ccdv/arxiv-summarization"]
    for item in dataset_name:
        dataset = load_dataset(item, "document", split="test")
        output_file = "result/try0/" + item.split("/")[-1] + ".json"
        split_summary(dataset, output_file)

def jsonl_to_json_and_csv(file_path: str, json_save_path: str, test_set_name: str, attack_name: str, if_save: dict):
    """
    convert jsonl file to json file
    """
    csv_save_path = json_save_path[:-5] + ".csv"
    with open(file_path, "r") as f:
        lines = f.readlines()
        json_data = []
        csv_data = []
        with tqdm(total=len(lines)) as pbar: # 进度条
            for line in lines:
                temp = json.loads(line)
                # modify id and add test_set_name and attack_name
                temp["id"] = str(temp["id"]) + "_" + test_set_name + "_" + attack_name
                temp["test_set_name"] = test_set_name
                temp["attack_name"] = attack_name
                json_data.append(temp)
                # save csv
                temp_csv = {
                    "premise": remove_return(temp["text"]),
                    "hypothesis": remove_return(temp["claim"]),
                    "label": temp["label"]
                }
                csv_data.append(temp_csv)
                # update progress bar
                pbar.update(1)
        # save json and csv
        if if_save["save_json"]:
            json_write(json_save_path, json_data)
        if if_save["save_csv"]:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_save_path, index=False, sep='\t') 
    return json_save_path, csv_save_path

def get_all_json_and_json_and_csv_from_jsonl():
    """
    generate json and csv for liu data and generate the json with all data
    """
    if_save = {
        "save_json": True,
        "save_csv": True,
    }
    data_root = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/base_and_diagnostic_sets"
    file_name = "data-dev.jsonl"
    addrs = [
        ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"],
        ["AntoSub", "NumEdit", "EntRep", "SynPrun"],
    ]

    data = []
    for path1 in addrs[0]:
        for path2 in addrs[1]:
            data_current = []
            file_path = data_root + "/" + path1 + "/" + path2 + "/" + file_name
            json_save_path, csv_save_path = jsonl_to_json_and_csv(file_path, file_path[:-6] + ".json", path1, path2, if_save)
            temp = json_read(json_save_path)
            for item in temp:
                data.append(item)
                data_current.append(item)
    json_write(data_root + "/adv_fact.json", data)

def json_to_jsonl(input_json: str, output_jsonl: str):
    json_input = json_read(input_json)
    jsonl_file_true = []
    jsonl_file_false = []
    for item in json_input:
        temp = {
            "id": item["id"],
            "text": item["summary"],
            "claim": item["sentence"]
        }
        jsonl_file_true.append(temp)
        temp = {
            "id": item["id"],
            "text": item["summary"],
            "claim": item["perturbed_sentence"]
        }
        jsonl_file_false.append(temp)
    
    # output_jsonl_true is the file name before the .json of the output jsonl file
    output_jsonl_true = ".".join(output_jsonl.split(".")[:-1]) + "_true.jsonl"
    output_jsonl_false = ".".join(output_jsonl.split(".")[:-1]) + "_false.jsonl"
    jsonl_write(output_jsonl_true, jsonl_file_true)
    jsonl_write(output_jsonl_false, jsonl_file_false)

def sample_jsonl_to_jsonl(n: int, f_in_path: str, f_out_path: str):
    f_in = open(f_in_path, "r")
    f_out = open(f_out_path, "w")
    for i in range(n):
        line = f_in.readline().strip()
        f_out.write(line + "\n")
    f_in.close()
    f_out.close()
    return

def sample_jsonl_main(n: int = 3):
    f_in_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_process/sample.jsonl"
    f_out_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_process/sample1.jsonl"
    check_file_exists(f_out_path)
    sample_jsonl_to_jsonl(3, f_in_path, f_out_path)

def jsonl_read(file: str):
    """
    read jsonl file and return a list of json objects
    """
    with open(file, "r") as f:
        lines = f.readlines()
        json_list = []
        for line in lines:
            temp = json.loads(line)
            json_list.append(temp)
    return json_list

def jsonl_write(jsonl_file: str, output_jsonl: list):
    """
    write output_jsonl to jsonl_file in jsonl format
    """
    with open(jsonl_file, "w") as f:
        for item in output_jsonl:
            temp = json.dumps(item)
            f.write(temp+"\n")
    return

def extract_jsonl_not_null_main(f_in_path: str = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_process/sample.jsonl"):
    """
    extract the jsonl file that augmentation is not null
    """
    jsonl = jsonl_read(f_in_path)
    not_null_sent = []
    error_sent = []
    for item in jsonl:
        if "augmentation" not in item.keys():
            error_sent.append(item)
        elif item["augmentation"] is not None:
            not_null_sent.append(item)
    f_out_path = '.'.join(f_in_path.split(".")[:-1]) + "_not_null.jsonl"
    f_error_path = '.'.join(f_in_path.split(".")[:-1]) + "_error.jsonl"
    jsonl_write(f_out_path, not_null_sent)
    jsonl_write(f_error_path, error_sent)
    print("length of not-null-sents:", len(not_null_sent))
    print("length of error-sents:", len(error_sent))

def json_to_csv(input_json: str, output_csv: str, keys_in: list=["summary", "sentence", "perturbed_sentence"]):
    # check if the given file exists or not, if not, create it. do not use json_check_file here
    if not os.path.exists(input_json):
        # create the file
        with open(input_json, "w") as f:
            json.dump([], f)

    json_input = json_read(input_json)
    csv_file = []
    for item in json_input:
        temp = {
            "premise": item[keys_in[0]],
            "hypothesis": item[keys_in[1]],
            "label": "CORRECT"
        }
        csv_file.append(temp)
        temp = {
            "premise": item[keys_in[0]],
            "hypothesis": item[keys_in[2]],
            "label": "CORRECT"
        }
        csv_file.append(temp)
    df = pd.DataFrame(csv_file)
    df.to_csv(output_csv, index=False, sep='\t') 
    return

def json_to_csv_34912(input_json: str, output_csv: str):
    # check if the given file exists or not, if not, create it. do not use json_check_file here
    if not os.path.exists(input_json):
        # create the file
        with open(input_json, "w") as f:
            json.dump([], f)

    json_input = json_read(input_json)
    csv_file = []
    for item in json_input:
        temp = {
            "premise": item["text"],
            "hypothesis": item["claim"],
            "label": item["label"]
        }
        csv_file.append(temp)
    df = pd.DataFrame(csv_file)
    df.to_csv(output_csv, index=False, sep='\t') 
    return

def jsonl_to_csv_34912_main():
    """
    from `text, claim, label`.json to `premise, hypothesis, label`.csv
    """
    input = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/data_process/sample.jsonl"
    sents = jsonl_read(input)
    csv_file = []
    for item in sents:
        temp = {
            "premise": item["text"],
            "hypothesis": item["claim"],
            "label": item["label"]
        }
        csv_file.append(temp)
    df = pd.DataFrame(csv_file)
    output_csv = '.'.join(input.split(".")[:-1]) + ".csv"
    df.to_csv(output_csv, index=False, sep='\t') 
    return

def temp():
    file_path = "/Users/gmh/Downloads/shell_script/file/our_data.json"
    file_path = "/Users/gmh/Downloads/shell_script/file/advfact_data.json"
    
    data = json_read(file_path)
    jsons = []
    # for i in range(3):
    #     jsons.append(data[i])
    for item in data:
        if item["label"] == "INCORRECT":
            jsons.append(item)
    output_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/_learn/temp.json"
    json_write(output_path, jsons)

def json_to_csv_advfact_data(input_json: str, output_csv: str):
    # check if the given file exists or not, if not, create it. do not use json_check_file here
    if not os.path.exists(output_csv):
        # create the file
        with open(output_csv, "w") as f:
            json.dump([], f)

    json_input = json_read(input_json)
    csv_file = []
    for item in json_input:
        temp = {
            "premise": remove_return(item["text"]),
            "hypothesis": remove_return(item["claim"]),
            "label": item["label"]
        }
        csv_file.append(temp)
    df = pd.DataFrame(csv_file)
    df.to_csv(output_csv, index=False, sep='\t') 
    return

def json_to_csv_advfact_data_main():
    input_json = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/rose_data/our_data.json"
    output_csv = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/rose_data/our_data.csv"
    json_to_csv_advfact_data(input_json, output_csv)
    return

def sample_json(n: int, file_path: str):
    data = json_read(file_path)
    data_sample = data[:n]
    output_path = ".".join(file_path.split(".")[:-1]) + "_sample.json"
    json_write(output_path, data_sample)

def sample_json_main():
    file_path = "/root/autodl-tmp/advFact/pa_adv/train/data_file/origin_json/advfact.json"
    # file_path = "/root/autodl-tmp/advFact/pa_adv/train/data_file/origin_json/ours_data.json"
    sample_json(20, file_path)

def sample_csv(n: int, file_path: str):
    data = pd.read_csv(file_path, sep='\t')
    data_sample = data[:n]
    output_path = ".".join(file_path.split(".")[:-1]) + "_sample.csv"
    data_sample.to_csv(output_path, index=False, sep='\t')

def sample_csv_main():
    file_path = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/rose_data/our_data.csv"
    sample_csv(100, file_path)

def check_labels_csv(file_path: str):
    data = pd.read_csv(file_path, sep='\t')
    labels = data["label"].tolist()
    abnormal_labels = []
    for item in labels:
        if item not in ["CORRECT", "INCORRECT"]:
            abnormal_labels.append(item)
    print(set(labels))
    print("abnormal_labels_length: \n", len(abnormal_labels))
    print("abnormal_labels: \n", abnormal_labels)

def check_labels_json(file_path: str):
    data = json_read(file_path)
    labels = []
    for item in data:
        labels.append(item["label"])
    abnormal_labels = []
    for item in data:
        if item["label"] not in ["CORRECT", "INCORRECT"]:
            abnormal_labels.append(item)
    print(set(labels))
    print("abnormal_labels_length: \n", len(abnormal_labels))
    print("abnormal_labels: \n", abnormal_labels)

def check_labels_main():
    file_path = "/Users/gmh/Downloads/shell_script/file/our_data.json"
    check_labels_json(file_path)

def txt_read_each(path: str):
    """
    if the txt file exists, read the txt file and return the list.
    if the txt file does not exist, return False
    """
    # check if the given file exists or not, if not, create it. do not use json_check_file here        
    txt = []
    txt_exist = os.path.exists(path)
    if txt_exist:
        with open(path, "r") as f:
            res = f.readlines()
            for item in res:
                temp = item.split("\t")[1:]
                temp[-1] = temp[-1].strip("\n")
                txt.append(temp)
    return txt_exist, txt

def get_num(res: list):
    """
    get the number of TT, TF, FT, FF
    """
    TT = 0
    TF = 0
    FT = 0
    FF = 0
    for item in res[1:]:
        if item[0] == "CORRECT" and item[1] == "CORRECT":
            TT += 1
        elif item[0] == "CORRECT" and item[1] == "INCORRECT":
            TF += 1
        elif item[0] == "INCORRECT" and item[1] == "CORRECT":
            FT += 1
        elif item[0] == "INCORRECT" and item[1] == "INCORRECT":
            FF += 1
    return TT, TF, FT, FF

def save_result(path: str, acc0: float, acc1: float, acc2: float, acc3: float, acc: float):
    with open(path, 'w') as f:
        f.write("""
    acc0 = FF / (FT + FF) = {}
    acc1 = TT / (TT + TF) = {}
    acc2 = FF / (TF + FF) = {}
    acc3 = TT / (TT + FT) = {}
    acc = (TT + FF) / (TT + TF + FT + FF) = {}
        """
                .format(acc0, acc1, acc2, acc3, acc))
    return

def txt_read(txt_path: str):
    eval = {}
    # print("txt_path:", txt_path)
    with open(txt_path, "r") as f:
        res = f.readlines()
        for item in res:
            temp = item.split("=")
            for i in range(len(temp)):
                temp[i] = temp[i].strip(" ")
            # print("temp[0]: ", temp[0])
            # print("temp[1]: ", remove_return(temp[1]))
            eval[temp[0]] = remove_return(temp[1])
    return eval

def cal_acc_and_save_for_train_test(file_info: dict, result_file_name: str):
    """
    calculate the acc from `file_info["eval_result_txt"]`.
    save the result to `result_file_name` in `file_info["result_path"]`.
    """
    file_path = file_info["eval_result_txt"]
    # read the txt file and get the list
    exist, res = txt_read_each(file_path)
    if not exist:
        return
    # get the acc
    TT, TF, FT, FF = get_num(res)
    print("TT: {}, TF: {}, FT: {}, FF: {}".format(TT, TF, FT, FF))
    acc0 = FF / (FT + FF) if (FT + FF) != 0 else -1
    acc1 = TT / (TT + TF) if (TT + TF) != 0 else -1
    acc2 = FF / (TF + FF) if (TF + FF) != 0 else -1
    acc3 = TT / (TT + FT) if (TT + FT) != 0 else -1
    acc = (TT + FF) / (TT + TF + FT + FF) if (TT + TF + FT + FF) != 0 else -1
    result_path = "/".join(file_path.split("/")[:-1]) + "/" + result_file_name
    save_result(result_path, acc0, acc1, acc2, acc3, acc)
    # print("acc = FF / (FT + FF) = {}".format(acc0))

    txt_info_ditc = {}
    txt_info_ditc = txt_read(file_info["eval_results_None"])
    txt_info_ditc["acc0"] = acc0
    txt_info_ditc["acc1"] = acc1
    txt_info_ditc["acc2"] = acc2
    txt_info_ditc["acc3"] = acc3
    txt_info_ditc["acc"] = acc
    return txt_info_ditc

def cal_acc_for_train_test_main():
    """
    calculate the accuracy of the txt files
    """
    addrs = [
        ["/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir"],
        ["our"], # "liu", 
        ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"],
        ["AntoSub", "NumEdit", "EntRep", "SynPrun"],
        ["bert_mnli", "electra_mnli", "roberta_mnli"], # "factcc_sub" 在这里添加数据集文件夹名称
        ["candidate.txt"]
    ]
    file_info = []
    for path1 in addrs[1]:
        dir1 = addrs[0][0] + "/" + path1
        for path2 in addrs[2]:
            dir2 = dir1 + "/" + path2
            for path3 in addrs[3]:
                dir3 = dir2 + "/" + path3
                for path4 in addrs[4]:
                    dir4 = dir3 + "/" + path4
                    path = dir4 + "/" + addrs[-1][0]
                    # result_txt = dir4 + "/candidate.txt"
                    eval_results_None = dir4 + "/eval_results_None.txt"

                    temp = {
                        "dir": dir4,
                        # "result_list_txt": path,
                        "eval_result_txt": path,
                        "eval_results_None": eval_results_None,
                        "liu_our": path1,
                        "dataset": path2,
                        "attack": path3,
                        "eval_set": path4
                    }
                    file_info.append(temp)

    result_file_name = "acc_results.txt"
    info_dict_list = []
    for item in file_info:
        info_dict = cal_acc_and_save(item, result_file_name)
        info_dict["liu_our"] = item["liu_our"]
        info_dict["dataset"] = item["dataset"]
        info_dict["attack"] = item["attack"]
        info_dict["eval_set"] = item["eval_set"]
        info_dict_list.append(info_dict)
    info_path = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir/contrast.json"
    json_write(info_path, info_dict_list)
    return

def cal_acc_and_save(file_info: dict, result_file_name: str):
    """
    calculate the acc from `file_info["eval_result_txt"]`.
    save the result to `result_file_name` in `file_info["result_path"]`.
    """
    file_path = file_info["eval_result_txt"]
    # read the txt file and get the list
    exist, res = txt_read_each(file_path)
    if not exist:
        return
    # get the acc
    TT, TF, FT, FF = get_num(res)
    # print("TT: {}, TF: {}, FT: {}, FF: {}".format(TT, TF, FT, FF))
    acc0 = FF / (FT + FF) if (FT + FF) != 0 else -1
    acc1 = TT / (TT + TF) if (TT + TF) != 0 else -1
    acc2 = FF / (TF + FF) if (TF + FF) != 0 else -1
    acc3 = TT / (TT + FT) if (TT + FT) != 0 else -1
    acc = (TT + FF) / (TT + TF + FT + FF) if (TT + TF + FT + FF) != 0 else -1
    result_path = "/".join(file_path.split("/")[:-1]) + "/" + result_file_name
    save_result(result_path, acc0, acc1, acc2, acc3, acc)
    # print("acc = FF / (FT + FF) = {}".format(acc0))

    txt_info_ditc = {}
    txt_info_ditc = txt_read(file_info["eval_results_None"])
    txt_info_ditc["acc0"] = acc0
    txt_info_ditc["acc1"] = acc1
    txt_info_ditc["acc2"] = acc2
    txt_info_ditc["acc3"] = acc3
    txt_info_ditc["acc"] = acc
    return txt_info_ditc

def cal_acc_main():
    # info_path = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir/contrast.json"
    output_dir = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir"
    info_path = output_dir + "/contrast.json"

    addrs = [
        [output_dir],
        ["liu", "our"],
        ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"],
        ["AntoSub", "NumEdit", "EntRep", "SynPrun"],
        ["bert_mnli", "electra_mnli", "roberta_mnli"], #, "factcc_sub"], # 在这里添加数据集文件夹名称
        ["", "", "candidate.txt"]
    ]
    cal_acc(addrs)

def cal_acc(addrs: list) -> str:
    """
    calculate the accuracy of the txt files
    """
    file_info = []
    # 一层一层的遍历文件夹
    for path1 in addrs[1]:
        dir1 = addrs[0][0] + "/" + path1
        for path2 in addrs[2]:
            dir2 = dir1 + "/" + path2
            for path3 in addrs[3]:
                dir3 = dir2 + "/" + path3
                for path4 in addrs[4]:
                    dir4 = dir3 + "/" + path4
                    path = dir4 + "/" + addrs[-1][2]
                    eval_results_None = dir4 + "/eval_results_None.txt"

                    temp = {
                        "dir": dir4,
                        "eval_result_txt": path,
                        "eval_results_None": eval_results_None,
                        "liu_our": path1,
                        "dataset": path2,
                        "attack": path3,
                        "eval_set": path4
                    }
                    file_info.append(temp) # 将所有文件的路径信息保存到file_info中，下面进行读取

    info_path = addrs[0][0] + "/file_info.json"
    print(">>> writing file_info:", info_path)
    json_write(info_path, file_info)

    result_file_name = "acc_results.txt"
    info_dict_list = []
    info_path = addrs[0][0] + "/contrast.json"
    json_write(info_path, info_dict_list)
    for item in file_info:
        # 遍历所有文件，汇总结果（之前还计算另一个东西，但是目前没啥用，就是cal_acc_and_save里面的计算）
        info_dict = cal_acc_and_save(item, result_file_name)
        # print("$$$ info_dict $$$", info_dict, info_dict.__class__)
        # print("$$$ item $$$", item, item.__class__)
        try:
            # 汇总并保存数据
            info_dict["liu_our"] = item["liu_our"]
            info_dict["dataset"] = item["dataset"]
            info_dict["attack"] = item["attack"]
            info_dict["eval_set"] = item["eval_set"]
            info_dict_list.append(info_dict)
            json_write(info_path, info_dict_list)
        except:
            raise Exception("{}".format(item))
    
    return info_path

def excel_write_from_json_v1(content: dict, save_path: str):
    metrics = ["bert_mnli", "electra_mnli", "roberta_mnli"] # , "factcc_mnli"
    evalsets = ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"]
    workbook = xlwt.Workbook(encoding= 'ascii')
    worksheet = workbook.add_sheet("result")
    acc_data = content

    for i in [0, 10]:
        worksheet.write(i + 0, 0, "Evaluation Set")
        worksheet.write(i + 1, 0, "Transf.")
        worksheet.write(i + 2, 0, metrics[0])
        worksheet.write(i + 3, 0, metrics[1])
        worksheet.write(i + 4, 0, metrics[2])
        worksheet.write(i + 0, 1, evalsets[0])
        worksheet.write(i + 0, 9, evalsets[1])
        worksheet.write(i + 0, 17, evalsets[2])
        worksheet.write(i + 0, 25, evalsets[3])
        worksheet.write(i + 0, 33, evalsets[4])
        worksheet.write(i + 0, 41, evalsets[5])
    for i in range(6):
        col = 1+i*8
        worksheet.write(1, col+0, "AntoSub")
        worksheet.write(1, col+2, "NumEdit")
        worksheet.write(1, col+4, "EntRep")
        worksheet.write(1, col+6, "SynPrun")
    for item in acc_data:
        row_base = 0
        if item["eval_set"] == metrics[0]:
            row = row_base + 2
        elif item["eval_set"] == metrics[1]:
            row = row_base + 3
        elif item["eval_set"] == metrics[2]:
            row = row_base + 4

        if item["dataset"] == evalsets[0]:
            col_base = 1
        elif item["dataset"] == evalsets[1]:
            col_base = 9
        elif item["dataset"] == evalsets[2]:
            col_base = 17
        elif item["dataset"] == evalsets[3]:
            col_base = 25
        elif item["dataset"] == evalsets[4]:
            col_base = 33
        elif item["dataset"] == evalsets[5]:
            col_base = 41

        if item["attack"] == "AntoSub":
            col = col_base + 0
        elif item["attack"] == "NumEdit":
            col = col_base + 2
        elif item["attack"] == "EntRep":
            col = col_base + 4
        elif item["attack"] == "SynPrun":
            col = col_base + 6

        if item["liu_our"] == "liu":
            col += 0 
        else:
            col += 1
        
        worksheet.write(row, col, item["eval_accuracy"])
        worksheet.write(row+10, col, item["acc0"])

    workbook.save(save_path)

def excel_write_from_2d_list(data: List[List[str]], save_path: str):
    """
    write a 2d list to excel file
    """
    output = open(save_path, 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            content = str(data[i][j])
            # print_hint_with_color(str(content.__class__), "yellow", str(content))
            if content == "None":
                output.write(" ")
            else:
                output.write(content)
            output.write('\t')
        output.write('\n')
    output.close()
    return

def read_excel(excel_path: str) -> List[List[str]]:
    """
    read excel file
    return a 2d list
    """
    data = pd.read_table(excel_path, header=None, nrows=100, encoding = "ISO-8859-1")
    matrix = data.values
    return matrix

def write_json_into_json_excel(json_path: str) -> Dict[str, list]:
    """
    write a json data into matrix
    return a info Dict[list]
    """
    updir = json_path.split("/")[-2]
    if "12" in updir:
        col_delta_1 = 0
    elif "34" in updir:
        col_delta_1 = 2
    elif "56" in updir:
        col_delta_1 = 4

    row_delta = 7
    col_delta = 7
    json_data = json_read(json_path)

    cells = []
    cells_dict = {}
    row_base = 4
    col_base = 2
    for item in json_data:        
        row = row_base
        col = col_base

        if item["attack"] == "AntoSub":
            col += col_delta * 0
        elif item["attack"] == "NumEdit":
            col += col_delta * 1
        elif item["attack"] == "EntRep":
            col += col_delta * 2
        elif item["attack"] == "SynPrun":
            col += col_delta * 3

        if item["dataset"] == "DocAsClaim":
            row += row_delta * 0
        elif item["dataset"] == "RefAsClaim":
            row += row_delta * 1
        elif item["dataset"] == "FaccTe":
            row += row_delta * 2
        elif item["dataset"] == "QagsC":
            row += row_delta * 3
        elif item["dataset"] == "RankTe":
            row += row_delta * 4
        elif item["dataset"] == "FaithFact":
            row += row_delta * 5

        if item["eval_set"] == "bert_mnli":
            row += 0
        elif item["eval_set"] == "electra_mnli":
            row += 1
        elif item["eval_set"] == "roberta_mnli":
            row += 2

        if item["liu_our"] == "liu":
            col += 0
        else:
            col += 1
        
        col += col_delta_1

        # excel_matrix[row][col] = item["eval_accuracy"]
        temp = {
            "row": row,
            "col": col,
            "value": item["eval_accuracy"]
        }
        cells.append(temp)
    cells_dict[json_path] = cells

    headers = []
    for i in range(6):
        row = row_base
        col = col_base - 1
        row += row_delta * i
        temp = {
            "row": row,
            "col": col,
            "value": "bert_mnli"
        }
        headers.append(temp)
        temp = {
            "row": row + 1,
            "col": col,
            "value": "electra_mnli"
        }
        headers.append(temp)
        temp = {
            "row": row + 2,
            "col": col,
            "value": "roberta_mnli"
        }
        headers.append(temp)
        for j in range(4):
            col_base_1 = col_base - 1 + col_delta * j
            temp = {
                "row": row - 1,
                "col": col_base_1 + 1,
                "value": "AdvFact(only test)"
            }
            headers.append(temp)
            temp = {
                "row": row - 1,
                "col": col_base_1 + 2,
                "value": "Ours(only test)"
            }
            headers.append(temp)
            temp = {
                "row": row - 1,
                "col": col_base_1 + 3,
                "value": "GPT3_test-AdvFact"
            }
            headers.append(temp)
            temp = {
                "row": row - 1,
                "col": col_base_1 + 4,
                "value": "GPT3_test-ours"
            }
            headers.append(temp)
            temp = {
                "row": row - 1,
                "col": col_base_1 + 5,
                "value": "sent_level-AdvFact"
            }
            headers.append(temp)
            temp = {
                "row": row - 1,
                "col": col_base_1 + 6,
                "value": "sent_level-ours"
            }
            headers.append(temp)
    for j in range(6):
        row = row_base - 2 + row_delta * j
        col = col_base
        temp = {
            "row": row,
            "col": col + col_delta * 0,
            "value": "AntoSub"
        }
        headers.append(temp)
        temp = {
            "row": row,
            "col": col + col_delta * 1,
            "value": "NumEdit"
        }
        headers.append(temp)
        temp = {
            "row": row,
            "col": col + col_delta * 2,
            "value": "EntRep"
        }
        headers.append(temp)
        temp = {
            "row": row,
            "col": col + col_delta * 3,
            "value": "SynPrun"
        }
        headers.append(temp)
    for i in range(1):
        row = row_base - 3
        col = col_base
        temp = {
            "row": row + row_delta * 0,
            "col": col,
            "value": "DocAsClaim"
        }
        headers.append(temp)
        temp = {
            "row": row + row_delta * 1,
            "col": col,
            "value": "RefAsClaim"
        }
        headers.append(temp)
        temp = {
            "row": row + row_delta * 2,
            "col": col,
            "value": "FaccTe"
        }
        headers.append(temp)
        temp = {
            "row": row + row_delta * 3,
            "col": col,
            "value": "QagsC"
        }
        headers.append(temp)
        temp = {
            "row": row + row_delta * 4,
            "col": col,
            "value": "RankTe"
        }
        headers.append(temp)
        temp = {
            "row": row + row_delta * 5,
            "col": col,
            "value": "FaithFact"
        }
        headers.append(temp)
    cells_dict["header"] = headers
    return cells_dict

def json_add_to_shared_excel(json_path: str, shared_excel_path: str):
    # matrix = read_excel(excel_path)
    excel_json_path = ".".join(shared_excel_path.split(".")[:-1]) + ".json"
    json_check_file(excel_json_path)
    excel_json = json_read(excel_json_path)
    if len(excel_json) > 0:
        excel_json = excel_json[0]
    else:
        excel_json = {}
    cells_dict = write_json_into_json_excel(json_path) # 
    excel_json[json_path] = []
    for item in cells_dict[json_path]:
        excel_json[json_path].append(item)
    excel_json["header"] = []
    for item in cells_dict["header"]:
        excel_json["header"].append(item)
    json_write(excel_json_path, [excel_json])
    cell_matrix = np.ndarray(shape=(200, 200), dtype=object)
    for path in excel_json:
        for cell in excel_json[path]:
            cell_matrix[cell["row"]][cell["col"]] = cell["value"]
    # cell_matrix = xls_header_add(cell_matrix)
    excel_write_from_2d_list(cell_matrix, shared_excel_path)
    # excel_write_from_2d_list(matrix, excel_path)
    return shared_excel_path

def json_to_excel_info(json_path: str) -> str:
    excel_path = ".".join(json_path.split(".")[:-1]) + ".xls"
    json_data = json_read(json_path)
    excel_write_from_json_v1(json_data, excel_path)
    return excel_path

def json_to_excel_info_main():
    # contrast_path = "/root/autodl-tmp/advFact/pa_adv/test/output_dir/contrast.json" # for test
    contrast_path = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir/contrast.json" # for train - test
    json_to_excel_info(contrast_path)

def shell_write_and_chmod(path: str, shell: str):
    """
    write shell to path and chmod +x
    """
    with open(path, "w") as f:
        f.write(shell)
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    return

def get_model_name_or_path(model_name_or_path: str, liu_our: str):
    """
    liu_our should be "liu" or "our" or "liu_google"
    model_name_or_path
    """
    info = json_read(loc_info_path)
    if (
        model_name_or_path in ["bert_mnli", "electra_mnli", "roberta_mnli"] and
        liu_our in ["our", "liu", "liu_name"]
    ):
        res = info["model"][liu_our][model_name_or_path]
    else:
        # for liu's model from google drive
        if liu_our == "liu_google":
            if model_name_or_path in ["bert_mnli", "electra_mnli", "factcc_sub"]:
                res = info["model"][liu_our][model_name_or_path]
            else:
                res = "not found"
        else:
            res = "not found"
    return res

def generate_shell_script_for_test_main():
    shell_info = {
        "python_file_path": "/root/autodl-tmp/advFact/pa_adv/baseline/roberta_bert_electra.py",
        "model_name_or_path": "",
        "output_dir": "/root/autodl-tmp/advFact/pa_adv/test/ours_data/sent_level",
        "validation_file": "",
        "train_file": "",
        "cache_dir": "/root/autodl-tmp/advFact/pa_adv/test/ours_data/cache_dir",
    }
    shell_info["train_file"]: shell_info["validation_file"]
    csv_files_to_be_tested_root_dir = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/sent_level/csv"
    csv_name = "advfact.csv"
    addrs = [
        ["output_dir", "csv_files_to_be_tested_root_dir"], # 0: output direct, 1: csv data direct
        ["liu", "our"],
        ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"],
        ["AntoSub", "NumEdit", "EntRep", "SynPrun"],
        ["bert_mnli", "electra_mnli", "roberta_mnli"],   #, "factcc_sub"], 在这里添加数据集文件夹名称
        ["test.sh", "csv_name"]  # 0: shell script name, 1: csv data name
    ]
    json_script_path = generate_shell_script_for_test(addrs, shell_info, csv_files_to_be_tested_root_dir, csv_name)
    return

def generate_shell_script_for_test(addrs: list, shell_info: dict, rewrite_shell_list: bool, model_liu_or_our = "our") -> str:
    """
    addrs: list of list
    shell_info: dict
    model_liu_or_our: "our" or "liu" or "liu_google"
    """
    script_name = "scripts.json" # 汇总的shell脚本的名字

    python_file_path = shell_info["python_file_path"]
    model_name_or_path = shell_info["model_name_or_path"]
    output_dir = shell_info["output_dir"]
    check_dir(output_dir)
    validation_file = shell_info["validation_file"]
    train_file = shell_info["train_file"]
    cache_dir = shell_info["cache_dir"]
    check_dir(cache_dir)

    # shell脚本
    shell_base = "CUDA_VISIBLE_DEVICES=0 python {python_file_path} --model_name_or_path {model_name_or_path} --do_eval --do_predict --max_seq_length 512 --output_dir {output_dir} --train_file {train_file} --validation_file {validation_file} --cache_dir {cache_dir}"

    script_list = [] # 汇总的shell脚本，之后写入json文件
    
    check_dir(addrs[0][0])
    # 一层一层的遍历目录，生成shell脚本
    for path1 in addrs[1]:
        dir1 = addrs[0][0] + "/" + path1
        csv_dir1 = addrs[0][1] + "/" + path1
        check_dir(dir1)
        for path2 in addrs[2]:
            dir2 = dir1 + "/" + path2
            csv_dir2 = csv_dir1 + "/" + path2
            check_dir(dir2)
            for path3 in addrs[3]:
                dir3 = dir2 + "/" + path3
                csv_dir3 = csv_dir2 + "/" + path3
                check_dir(dir3)
                for path4 in addrs[4]:
                    dir4 = dir3 + "/" + path4
                    check_dir(dir4)
                    shell_path = dir4 + "/" + addrs[-1][0]
                    validation_file = csv_dir3 + "/" + addrs[-1][1]
                    model_name_or_path = get_model_name_or_path(path4, model_liu_or_our)
                    if model_name_or_path == "not found":
                        print_hint_with_color(">>> model_name_or_path `{}` not found, continue".format(model_liu_or_our), "red")
                        continue
                    shell = shell_base.format(
                        python_file_path = python_file_path,
                        model_name_or_path = model_name_or_path,
                        output_dir = dir4,
                        validation_file = validation_file,
                        train_file = validation_file,
                        cache_dir = cache_dir
                    )

                    temp = {
                        "shell_script": shell,
                        "shell_path": shell_path,
                        "shell_dir": dir4,
                        "model_path": model_name_or_path,
                        "output_dir": dir4,
                        "train_file": validation_file,
                        "validation_file": validation_file,
                        "executed": False,
                        "result": None
                    }
                    script_list.append(temp)
    json_script_path = addrs[0][0] + "/" + script_name
    json_list_exist = check_file_exists(json_script_path)
    # print_hint_with_color("json_script_path", "cyan", ": {}".format(json_script_path))
    # print_hint_with_color("rewrite_shell_list", "cyan", ": True" if rewrite_shell_list else ": False")
    # print_hint_with_color("json_list_exist", "cyan", ": True" if json_list_exist else ": False")
    # ask_if_continue()

    # 是否覆盖掉之前的shell汇总文件（该文件用于遍历执行shell脚本，所以很关键）
    if not json_list_exist:
        # print_hint_with_color("writing script_list", "cyan")
        # ask_if_continue()
        json_write(json_script_path, script_list)
    else:
        if rewrite_shell_list:
            print_hint_with_color("rewriting script_list", "cyan")
            # ask_if_continue()
            json_write(json_script_path, script_list)
        else:
            script_list = json_read(json_script_path)

    # ask_if_continue()
    # check if the given dir exists or not, if not, create it
    for item in script_list:
        if not os.path.exists(item["shell_dir"]):
            os.makedirs(item["shell_dir"])
        # write shellscript into shell_path
        check_file_exists(item["shell_path"])
        # 写脚本，同时修改权限
        shell_write_and_chmod(item["shell_path"], item["shell_script"])
    return json_script_path

def check_and_mkdir(dir_path: str):
    """
    check if the given file `dir_path` exists or not.
    if exist, return True.
    if not, create it and return False.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return False
    return True

def json_to_csv_advfact_data_different_kind(input_json: str, output_dir: str, file_name: str):
    """
    given the original json file, convert it into [eval-set] x [attack] csv files in corresponding dir
    """
    save_json_version = True # save the json version of the csv file at the same time 便于查看，csv看起来难受，json看起来舒服

    json_input = json_read(input_json)
    # 文件目录
    csv_file = {
        "DocAsClaim": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
        "RefAsClaim": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
        "FaccTe": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
        "QagsC": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
        "RankTe": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
        "FaithFact": {
            "AntoSub": [],
            "NumEdit": [],
            "EntRep": [],
            "SynPrun": [],
        },
    }

    for item in json_input:
        # remove_return 去掉文本中的换行符
        temp = {
            "premise": remove_return(item["text"]),
            "hypothesis": remove_return(item["claim"]),
            # "hypothesis": remove_return(item["origin_claim"]),
            "label": item["label"]
        }
        # csv_file是一个2d数组，每个元素是一个list，list中是一些dict，形如上面的temp
        csv_file[item["test_set_name"]][item["attack_name"]].append(temp)

    # make dir of the output_dir and the csv_file
    # write csv file
    check_and_mkdir(output_dir)
    with tqdm(total=len(csv_file)) as pbar_0:
        for test_set in csv_file:
            print_hint_with_color("\n" + test_set, "cyan", ": {}".format(test_set))
            # 进入test_set对应的目录
            temp_path = output_dir + "/" + test_set if output_dir[-1] != "/" else output_dir + test_set
            check_and_mkdir(temp_path)
            for attack in csv_file[test_set]:
                # 进入attack对应的目录
                temp_path_path = temp_path + "/" + attack if temp_path[-1] != "/" else temp_path + attack
                check_and_mkdir(temp_path_path)
                output_csv_file_path = temp_path_path + "/" + file_name if temp_path_path[-1] != "/" else temp_path_path + file_name
                check_file_exists(output_csv_file_path)
                # 写入csv文件
                csv_json_file = output_dir + "/csv_json_file.json"
                json_check_file(csv_json_file)
                json_write(csv_json_file, csv_file)
                df = pd.DataFrame(csv_file[test_set][attack])
                df.to_csv(output_csv_file_path, index=False, sep='\t') 
                if save_json_version:
                    # 同时保存一份json版本的文件，方便查看
                    json_output = ".".join(output_csv_file_path.split(".")[:-1]) + ".json"
                    json_write(json_output, csv_file[test_set][attack])
            pbar_0.update(1)
    return

def json_to_csv_advfact_data_different_kind_different_author(input_json_list: list, output_dir: str, file_name: str):
    """
    input_json_list: list of json file path. There shoule only be 2 json files in the list.
    output_dir: the dir to store the csv files. It is the parent dir of `liu` dir and `our` dir.
    file_name: the name of the csv file.
    """
    check_dir(output_dir)
    for json_file_path in input_json_list:
        # get the name from the file path
        json_file_name = json_file_path.split("/")[-1]
        # 找到output_dir下一层文件夹应该是liu还是our
        if "our" in json_file_name:
            output_dir_temp = output_dir + "/our"
            json_to_csv_advfact_data_different_kind(json_file_path, output_dir_temp, file_name)
        else:
            # output_dir_temp = output_dir + "/liu"
            cp_liu_files_to_dir(output_dir, "liu")
    return

def json_to_csv_advfact_data_different_kind_main():
    """
    generate csv file for different kind of advfact data in cooresponding file.
    `ours_data` is used for test
    """
    input_json = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/adv_and_our_data/origin_json/ours_data.json"
    output_dir = "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/adv_and_our_data/csv/our"
    file_name = "advfact.csv"
    json_to_csv_advfact_data_different_kind(input_json, output_dir, file_name)
    return

def shell_from_json(json_file: str, exec_log_path: str):
    """
    given the json file, execute the shell script in the json file.
    """
    shell_list = json_read(json_file) # 读取之前的shell汇总文件
    # log = shell_list.copy()
    with tqdm(total=len(shell_list)) as pbar: # 进度条
        for shell in shell_list:
            if not shell["executed"] or shell["result"] != 0:
                # 没执行过 或者 执行过但是结果不为0（0表示执行成功），就执行
                print(">>> executing shell: " + shell["shell_path"])
                res = os.system(shell["shell_path"])
                shell["executed"] = True
                shell["result"] = res
                if res != 0:
                    # 如果执行失败，记录下来
                    log = json_read(exec_log_path)
                    if "error_shell" not in log:
                        log["error_shell"] = []
                    error_shell = log["error_shell"]
                    error_shell.append(shell)
                    json_write(exec_log_path, log)
            # log.append(shell)
            # json_write(json_file, log)
            # shell_list[shell] = shell
            json_write(json_file, shell_list) # 保存执行结果
            pbar.update(1)
    return

def shell_from_json_main():
    # path = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir/scripts.json"
    json_script_path = ""
    path = json_script_path
    path = "/root/autodl-tmp/advFact/pa_adv/test/ours_data/output_dir/scripts_liu.json"
    shell_from_json(path)
    return

def split_dataset_json_to_csv(json_file: str):
    """
    split the dataset into train, valid and test
    """
    split_ratio = (8000, 2000, 1) # ratio of train: valid: test
    json_datas = json_read(json_file)
    datas = []
    for item in json_datas:
        temp = {
            "premise": remove_return(item["text"]),
            "hypothesis": remove_return(item["claim"]),
            "label": item["label"]
        }
        datas.append(temp)
    random.shuffle(datas)
    sets = {
        "train": [],
        "valid": [],
        "test": []
    }
    sets["train"] = datas[:int(len(datas) * split_ratio[0] / sum(split_ratio))]
    sets["valid"] = datas[int(len(datas) * split_ratio[0] / sum(split_ratio)):int(len(datas) * (split_ratio[0] + split_ratio[1]) / sum(split_ratio))]
    sets["test"] = datas[int(len(datas) * (split_ratio[0] + split_ratio[1]) / sum(split_ratio)):]

    path_dir = ".".join(json_file.split(".")[:-1])
    res = check_dir(path_dir)
    print("path_dir exist: ", res, path_dir)
    for item in sets:
        file_path = path_dir + "/" + item + ".csv"
        check_file_exists(file_path)
        print(">>> file_path: ", file_path)
        df = pd.DataFrame(sets[item])
        df.to_csv(file_path, index=False, sep='\t') 
    return

def split_dataset_json_to_csv_main():
    """
    the main function for split_dataset_json_to_csv
    """
    json_file = "/root/autodl-tmp/advFact/pa_adv/train/data_file/advfact_data.json"
    split_dataset_json_to_csv(json_file)

def generate_shell_for_train(train_dir_root: str, dataset_dir: str, models: dict, stdout: str = "stdout"):
    """
    generate shell script for train
    """
    python_file = "/root/autodl-tmp/advFact/pa_adv/baseline/roberta_bert_electra.py"
    output_dir_base = train_dir_root + "/output_dir"
    train_file = dataset_dir + "/train.csv"
    validation_file = dataset_dir + "/valid.csv"
    cache_dir = train_dir_root + "/cache_dir"
    for path in [output_dir_base, cache_dir]:
        check_dir(path)

    shell_base = "CUDA_VISIBLE_DEVICES=0,1,2,3 python {python_file} --model_name_or_path {model_name_or_path} --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 15 --output_dir {output_dir} --train_file {train_file} --validation_file {validation_file} --cache_dir {cache_dir}"

    shell_addr_base = train_dir_root + "/script"

    scripts = []
    for model in models:
        output_dir = output_dir_base + "/" + model
        model_name_or_path = models[model]
        if stdout == "stdout":
            addition = ""
        else:
            txt_path = shell_addr_base + "/stdout/" + model + ".txt"
            # get the parent directory of txt_path
            parent_dir = "/".join(txt_path.split("/")[:-1])
            check_dir(parent_dir)
            check_file_exists(txt_path)
            addition = " > " + txt_path
        shell = shell_base.format(python_file=python_file, model_name_or_path=model_name_or_path, output_dir=output_dir, train_file=train_file, validation_file=validation_file, cache_dir=cache_dir)
        shell_path = shell_addr_base + "/" + model + "_train.sh"

        shell_main_path = shell_addr_base + "/" + model + "_main.sh"
        shell_main = "sh " + shell_path + addition

        temp = {
            "shell_path": shell_path,
            "shell": shell,
            "model": model,
            "model_name_or_path": model_name_or_path,
            "shell_main": shell_main,
            "shell_main_path": shell_main_path
        }
        scripts.append(temp)

        # write the shell into shell_path
        with open(shell_path, "w") as f:
            f.write(shell)
        st = os.stat(shell_path)
        os.chmod(shell_path, st.st_mode | stat.S_IEXEC)

        shell_write_and_chmod(shell_main_path, shell_main)
        # with open(shell_main_path, "w") as f:
        #     f.write(shell_main)
        # st = os.stat(shell_main_path)
        # os.chmod(shell_main_path, st.st_mode | stat.S_IEXEC)

    return scripts

def train_main():
    """
    the main function for train
    """
    models = {
        "mnlibert": "bert-base-cased",
        "mnlielectra": "google/electra-base-discriminator", 
        "mnliroberta": "roberta-base"
    }
    train_dir_root = "/root/autodl-tmp/advFact/pa_adv/train/adv_fact"
    dataset_dir = "/root/autodl-tmp/advFact/pa_adv/train/data_file/advfact_data"
    scripts = generate_shell_for_train(train_dir_root=train_dir_root, dataset_dir=dataset_dir, models=models, stdout="file")
    script_json_path = train_dir_root + "/script.json"
    json_write(script_json_path, scripts)
    print_hint_with_color("    scripts: ", "cyan", script_json_path)
    log = []
    with tqdm(total=len(scripts)) as pbar:
        for shell in scripts:
            print_hint_with_color("current shell: ", "cyan", str(shell))
            res = os.system(shell["shell_main_path"])
            shell["result"] = res
            log.append(shell)
            pbar.update(1)

    log_path = train_dir_root + "/log.json"
    json_check_file(log_path)
    json_write(log_path, log)
    return

def get_files_from_dir(dir: str) -> List[str]:
    """
    given dir path, return a list of file names in the dir
    """
    files = []
    for file in os.listdir(dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        files.append(file)
    return files

def get_abs_path_of_jsons_from_dir(dir: str) -> List[str]:
    """
    given dir path, return a list of json file names in the dir
    """
    files = get_files_from_dir(dir)
    json_files = []
    for file in files:
        if file.endswith(".json"):
            json_files.append(dir + "/" + file)
    return json_files

def cp_liu_files_to_dir(dir: str, folder_name: str):
    """
    copy liu's files to dir
    """
    info = json_read(loc_info_path)
    liu_files = info["data"]["liu_files"]
    check_and_mkdir(dir)
    shells = [
        "cp -r " + liu_files + " " + dir,
        "mv " + dir + "/" + liu_files.split("/")[-1] + " " + dir + "/" + folder_name
        ]
    # todo
    # 1. cp
    # 2. mv
    reses = []
    for shell in shells:
        print_hint_with_color("    shell: ", "cyan", shell)
        res = os.system(shell)
        reses.append(res)
    return reses

def test(exec_list: map, paths: map):
    """
    the main function for test
    """
    # test中共有5个部分，这里从exec_list获取要执行哪几个部分
    # funcs[part_name]["execute"]表示是否执行
    # funcs[part_name]["func"]表示执行的函数
    funcs = {
        "json_to_csv_advfact_data_different_kind_different_author": {
            "execute": exec_list["json_to_csv_advfact_data_different_kind_different_author"],
            "func": json_to_csv_advfact_data_different_kind_different_author
        },
        "generate_shell_script_for_test": {
            "execute": exec_list["generate_shell_script_for_test"],
            "func": generate_shell_script_for_test
        },
        "shell_from_json": {
            "execute": exec_list["shell_from_json"],
            "func": shell_from_json
        },
        "cal_acc": {
            "execute": exec_list["cal_acc"],
            "func": cal_acc
        },
        "json_to_excel_info": {
            "execute": exec_list["json_to_excel_info"],
            # "func": json_to_excel_info
            "func": json_add_to_shared_excel
        }
    }

    # 从paths中获取各个重要路径，含义详见 Readme - 函数说明 - __main__ - test_main_12
    # datas to be tested
    input_json_file_root = paths["input_json_file_root"]
    # root dir of all the data in different folders
    file_folder_root = paths["file_folder_root"]
    check_dir(file_folder_root)
    # model to use
    model_liu_or_our = paths["model_liu_or_our"]
    # cache dir
    cache_dir = paths["cache_dir"]
    # xls_path 最终xls文件的存放路径
    xls_path = paths["xls_path"]
    check_file_exists(xls_path)

    # log路径，debug和查看运行情况使用
    exec_log_path = file_folder_root + "/exec_log.json"
    log_exist = json_check_file(exec_log_path)
    if log_exist:
        with open(exec_log_path, "r") as f:
            exec_log = json.load(f)
    else:
        exec_log = {
            "shell_exec_started": False,
            "shell_exec_finished": False,
            "json_script_path": paths["file_folder_root"] + "/json_script.json",
            "rewrite_json_list": True,  # 是否运行生成新的脚本汇总文件
        }
        json_write(exec_log_path, exec_log)
    if exec_log["shell_exec_started"]:
        rewrite_json_list = False # 如果已经开始执行shell，就不要再重写json_list了，否则会丢失信息，不知道哪些运行了，哪些还没运行（毕竟几乎所有耗时都在这个部分）
        exec_log["rewrite_json_list"] = rewrite_json_list
        json_write(exec_log_path, exec_log)

    # json_to_csv_advfact_data_different_kind_main() ############################## part 1
    # 生成不同文件夹下的csv文件
    json_test_file_list = get_abs_path_of_jsons_from_dir(input_json_file_root)
    if funcs["json_to_csv_advfact_data_different_kind_different_author"]["execute"]:
        # 执行json_to_csv_advfact_data_different_kind_different_author()函数
        funcs["json_to_csv_advfact_data_different_kind_different_author"]["func"](json_test_file_list, file_folder_root, "advfact.csv")
        print_hint_with_color("func done", "green", ": json_to_csv_advfact_data_different_kind_main()")

    # generate_shell_script_for_test_main() ############################## part 2
    # 填入shell中的关键信息
    shell_info = {
        "python_file_path": "/root/autodl-tmp/advFact/pa_adv/baseline/roberta_bert_electra.py",
        "model_name_or_path": "",
        "output_dir": file_folder_root,
        "validation_file": "",
        "train_file": "",
        "cache_dir": "",
    }
    shell_info["cache_dir"] = cache_dir if len(cache_dir) != 0 else file_folder_root + "/cache_dir" # 若指定了cache_dir，则使用指定的，否则使用默认新建的空文件夹
    shell_info["train_file"] = shell_info["validation_file"] # 这里的train_file没有用，不知道为什么刘鹏飞的readme中这么写，之前测试过train_file写成别的，结果不变
    csv_files_to_be_tested_root_dir = file_folder_root
    csv_name = "advfact.csv" # 生成的csv文件名
    # addrs模拟file_folder_root下的文件目录树，addrs[0][0]为root，addr[1]为子文件夹，addr[2]为二级子文件夹，addr[3]为三级子文件夹···addrs[-1]为文件名
    addrs = [
        [file_folder_root, csv_files_to_be_tested_root_dir], # 0: output direct, 1: csv data direct
        ["liu", "our"],
        ["DocAsClaim", "RefAsClaim", "FaccTe", "QagsC", "RankTe", "FaithFact"],
        ["AntoSub", "NumEdit", "EntRep", "SynPrun"],
        ["bert_mnli", "electra_mnli", "roberta_mnli"],   #, "factcc_sub"], dir names of datasets
        ["test.sh", csv_name, "candidate.txt"]  # 0: shell script name, 1: csv data name, 2: result file name，之后需要从这里读取运行结果
    ]
    if funcs["generate_shell_script_for_test"]["execute"]:
        # 检查是否需要覆盖掉之前生成的shell汇总文件
        exec_log = json_read(exec_log_path)
        rewrite_json_list = exec_log["rewrite_json_list"]
        # 执行
        json_script_path = funcs["generate_shell_script_for_test"]["func"](addrs, shell_info, rewrite_json_list, model_liu_or_our)
        # 保存执行信息
        exec_log = json_read(exec_log_path)
        exec_log["json_script_path"] = json_script_path
        json_write(exec_log_path, exec_log)
        print_hint_with_color("func done", "green", ": generate_shell_script_for_test_main()")

    # shell_from_json_main() ############################## part 3
    if funcs["shell_from_json"]["execute"]:
        # 保存执行信息
        exec_log["shell_exec_started"] = True
        json_write(exec_log_path, exec_log)
        exec_log = json_read(exec_log_path)
        json_script_path = exec_log["json_script_path"] # 从执行信息中读取json_script_path（所有shell的汇总文件，里面包含shell命令，shell位置，是否执行过，运行结果等信息）
        funcs["shell_from_json"]["func"](json_script_path, exec_log_path)
        print_hint_with_color("func done", "green", ": shell_from_json_main()")
        # 保存执行信息
        exec_log["shell_exec_finished"] = True
        json_write(exec_log_path, exec_log)

    # cal_acc_main() ############################## part 4 
    print(" >>> func start: cal_acc_main()")
    addrs[0][0] = file_folder_root
    if funcs["cal_acc"]["execute"]:
        contrast_json_path = funcs["cal_acc"]["func"](addrs)
        # 保存执行信息
        exec_log = json_read(exec_log_path)
        exec_log["contrast_json_path"] = contrast_json_path
        json_write(exec_log_path, exec_log)
        print_hint_with_color("func done", "green", ": cal_acc_main()")

    # json_to_excel_info_main ############################## part 5
    if funcs["json_to_excel_info"]["execute"]:
        # 读取执行信息
        exec_log = json_read(exec_log_path)
        contrast_json_path = exec_log["contrast_json_path"]
        # excel_path = funcs["json_to_excel_info"]["func"](contrast_json_path)
        # 执行，写入xls文件
        excel_path = funcs["json_to_excel_info"]["func"](contrast_json_path, xls_path)
        # 保存执行信息
        exec_log = json_read(exec_log_path)
        exec_log["excel_path"] = excel_path
        json_write(exec_log_path, exec_log)
        print_hint_with_color("func done", "green", ": json_to_excel_info_main()")
        print_hint_with_color("     result excel: ", "green", excel_path)

    return

def try_func(func, *kargs):
    try:
        func(kargs[0], kargs[1])
    except Exception as err:
        print_hint_with_color("error\n", "red", traceback.format_exc())
    else:
        print_hint_with_color("done", "green", "")
    finally:
        print_hint_with_color("----------"*7, "blue", "\n")
    return

def test_test():
    """
    col of B and C
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": True,
        "generate_shell_script_for_test": True,
        "shell_from_json": False,
        "cal_acc": False,
        "json_to_excel_info": False,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_12",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_test",
        # model to use
        "model_liu_or_our": "liu",
        # cache dir
        "cache_dir": "",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])

def test_dataset_12():
    """
    col of B and C
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": True,
        "generate_shell_script_for_test": True,
        "shell_from_json": False,
        "cal_acc": False,
        "json_to_excel_info": False,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_12_from_liu",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_12_from_liu",
        # model to use
        "model_liu_or_our": "liu",
        # cache dir
        "cache_dir": "",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])

def test_dataset_56():
    """
    col of B and C
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": True,
        "generate_shell_script_for_test": True,
        "shell_from_json": True,
        "cal_acc": True,
        "json_to_excel_info": True,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_12_from_liu",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_56_from_liu",
        # model to use
        "model_liu_or_our": "liu_google",
        # cache dir
        "cache_dir": "",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])


def test_main_12():
    """
    col of B and C
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": False,
        "generate_shell_script_for_test": False,
        "shell_from_json": True,
        "cal_acc": False,
        "json_to_excel_info": False,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_12",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_12_origin",
        # model to use
        "model_liu_or_our": "liu",
        # cache dir
        "cache_dir": "",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])

def test_main_34():
    """
    col of D and E
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": False,
        "generate_shell_script_for_test": False,
        "shell_from_json": True,
        "cal_acc": True,
        "json_to_excel_info": True,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_12",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_34",
        # model to use
        "model_liu_or_our": "our",
        # cache dir
        "cache_dir": "",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])

def test_main_56():
    """
    col of F and G
    """
    exec_list = {
        "json_to_csv_advfact_data_different_kind_different_author": True,
        "generate_shell_script_for_test": True,
        "shell_from_json": True,
        "cal_acc": True,
        "json_to_excel_info": True,
    }
    paths = {
        # datas to be tested
        "input_json_file_root": "/root/autodl-tmp/advFact/pa_adv/test/files_downloaded/data_cent_level",
        # root dir of all the data in different folders
        "file_folder_root": "/root/autodl-tmp/advFact/pa_adv/test/output/data_56",
        # model to use
        "model_liu_or_our": "liu",
        # cache dir
        "cache_dir": "/root/autodl-tmp/advFact/pa_adv/test/ours_data/data_12/cache_dir",
        # xls_path
        "xls_path": "/root/autodl-tmp/advFact/pa_adv/test/output/result.xls"
    }
    try_func(test, *[exec_list, paths])

def train():
    # 执行这个两个函数即可，但是因为就train了没几次，最后汇总成一个函数的时候没确定是否可行
    split_dataset_json_to_csv_main()
    train_main()
    return


if __name__ == "__main__":
    # test
    
    # test_test()
    test_dataset_12()
    # test_dataset_56()

    # test_main_12()
    # test_main_34()
    # test_main_56()

