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
from multiprocessing import Process
from multiprocessing import Array
from multiprocessing import RLock, Lock, Event, Condition, Semaphore
import xlwt
# from func.temp import *
from interruptingcow import timeout

loc_info_path = ""
openai_threshold = 12000

# from utils.read_api_key import read_api_keys

# api_list = [
#     # "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",
#     # "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",
#     # "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",
#     # above keys are unavailable
#     "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",
#     "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",
#     "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",
#     # "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",
#     # "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",
#     # "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",
#     # below keys will be used in the future
#     # "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",
#     # "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",
#     # "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",
#     # "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",
#     # "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",
#     # "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",
#     ]
api_list = [
    # "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", 
    # "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"
    ]

def if_continue(question: str = ""):
    """
    ask a question
    """
    print_hint_with_color("Question", "yellow", " " + question)
    a = input("Answer yes or no: ")
    if a == "yes":
        return True
    else:
        return False

def print_hint_with_color(hint: str, color: str, content = ""):
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
    content = str(content)
    print(color_dict[color] + hint + color_dict["end"] + content)
    return

def json_read(filePath: str):
    """
    read a json file
    """
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data

def json_write(filePath: str, data):
    """
    write a list of string to a json file
    """
    file_exist_check_and_create(filePath)
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=4)

def file_exist_check(filePath: str):
    """
    check if a file exists
    """
    if os.path.exists(filePath):
        return True
    else:
        return False

def file_exist_check_and_create(filePath: str):
    """
    check if a file exists
    if not, create it, and return False
    if yes, return True
    """
    if file_exist_check(filePath):
        return True
    else:
        open(filePath, 'w').close()
        return False

def generate_sentences(num: int, type_pn: str,  datasetName = "cnn_dailymail", version = "3.0.0", split = "train") -> dict:
    """
    Generate a dictionary of `num` items from the dataset.
    """
    loc_info = json_read(loc_info_path)
    prompt_info = json_read(loc_info["root"] + loc_info["prompt"])
    error_types_all = prompt_info["gpt35"][type_pn]["error_types_all"]
    dataset = load_dataset(datasetName, version, split=split)
    indexes = random.sample(range(0, dataset.num_rows), num)
    picked_sents = []
    for i in indexes:
        data = dataset[i]
        claim = remove_return(data["highlights"])
        text = remove_return(data["article"])
        claim_sents = sent_tokenize(claim)
        index = random.randint(0, len(claim_sents)-1)
        claim_sent = claim_sents[index]
        temp = {
            "status": "not_started",   # "not_started", "in_progress", "finished", "unfinished"
            "text": text,
            "claim": claim,
            "claim_sent": claim_sent,
            "result": {}
        }
        for err_type in error_types_all:
            tt = {
                "finished": False, 
                "err_type": err_type,
                "orgn_sent": claim,
                "pert_sent": "",
                "response": "",
                "elapsed_time": -1
            }
            temp["result"][err_type] = tt
        picked_sents.append(temp)
    return picked_sents

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    return sent.replace("\n", " ")

def prepare_original_sents(quantity: int, output_json_path: str, type_pn: str) -> str:
    """
    prepare the original sentences
    return the path of the json file
    """
    sents = generate_sentences(quantity, type_pn, datasetName = "cnn_dailymail", version = "3.0.0", split = "train")
    json_write(output_json_path, sents)
    return output_json_path

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    return sent.replace("\n", " ")

def get_types(*args):
    """
    get the types of the assistant
    """
    loc_info = json_read(loc_info_path)
    prompt_path = loc_info["root"] + loc_info["prompt"]
    msg_dict = json_read(prompt_path)
    # find the layer in the json hierarchy
    for arg in args:
        msg_dict = msg_dict[arg]
    # get the len of the assistant
    if msg_dict["cases"][0]["used_for"] == "types_used":
        assis_types = list(msg_dict["cases"][0]["assistant"].keys())
    else:
        examples = msg_dict["cases"]
        for example in examples:
            if example["used_for"] == "types_used":
                assis_types = list(example["assistant"].keys())
                break

    # if "structure" in assis_types:
    #     assis_types.remove("structure")
    return assis_types    

def get_my_sents() -> list:
    sents = [
        "Jeffrey Pyne, 22, charged with first-degree murder after mother's body found at Michigan home in May 2011 - but denies hurting her .",
        "James Watson was jailed yesterday after causing the death of Brendon Main, 18, in a smash in Aberdeenshire in July 2011 .",
        "Stephanie Wilson, 28, found the letter in September 2012 after purchasing boots at Saks Fifth Avenue .",
        "Cengiz Nuray, 17, vanished in April 1995 after leaving his belongings behind on a beach in Santa Cruz and going for a walk alone .",
        "Hayley Sanders, now 23, developed an infection after the birth of her son, Jayden, at Birmingham's Heartlands Hospital in December 2009 .",
    ]
    return sents

def get_input_messages(*args) -> list:
    loc_info = json_read(loc_info_path)
    prompt_path = loc_info["root"] + loc_info["prompt"]
    msg_dict = json_read(prompt_path)
    # find the layer in the json hierarchy
    for arg in args:
        msg_dict = msg_dict[arg]

    args = list(args)
    assis_types = get_types(*args)
    type_num = len(assis_types)
    # get the system message content
    system_content = msg_dict["system"]["requirement"].format(type_num)

    msg = []
    temp = {
        "role": "system",
        "content": system_content
    }
    msg.append(temp)
    # print_hint_with_color("msg", "cyan")
    # for m in msg:
    #     print(m)

    for item in msg_dict["cases"]:
        if item["in_use"] == False:
            continue
        content = item["user"]
        temp = {
            "role": "user",
            "content": content
        }
        msg.append(temp)
        assistant_content = item["assistant"]
        sents = []
        num = 0
        for key in item["assistant"]:
            if key in assis_types:
                num += 1
                sents.append(item["assistant"][key])
        assistant_content = ""
        for i in range(num):
            assistant_content += f"[{i}]{sents[i]}\n"
        temp = {
            "role": "assistant",
            "content": assistant_content
        }
        msg.append(temp)
    return msg

def xls_write(json_path):
    xls_path = json_path.replace(".json", ".xls")
    content = json_read(json_path)
    file_exist_check_and_create(xls_path)

    workbook = xlwt.Workbook(encoding= 'ascii')
    worksheet = workbook.add_sheet("result")

    row_base = 1
    row_interval = 18
    row_interval = len(content[0]["perturbed_sent"]) + 2
    col_base = 1
    i = 0
    for item in content:
        if len(content[0]["perturbed_sent"]) == 2:
            if item["perturbed_sent"]["location"] == "not_valid":
                continue
        original_sent_ = item["original_sent_"]
        perturbed_sent = item["perturbed_sent"]
        row = row_base + row_interval * i
        col = col_base
        worksheet.write(row, col, "original")
        worksheet.write(row, col+1, original_sent_)
        row += 1
        for p_type in perturbed_sent:
            if p_type == "response":
                continue
            worksheet.write(row, col, p_type)
            worksheet.write(row, col+1, perturbed_sent[p_type])
            row += 1
        i += 1


    workbook.save(xls_path)

def get_input_msg(sent_to_be_uploaded: str, to_gnr_err_type: str, type_pn: str):
    loc_info_path = os.getcwd() + "/loc_info.json"
    loc_info = json_read(loc_info_path)
    prompt_path = loc_info["root"] + loc_info["prompt"]
    prompt_info = json_read(prompt_path)

    pn_info = prompt_info["gpt35"][type_pn]

    error_types_all = pn_info["error_types_all"]
    specific_error_types = pn_info["specific_error_types"]

    sent_prompt = pn_info["sent_prompt"]
    specific_prompt = pn_info["specific_prompt"]

    response_format = pn_info["format"]
    cases = pn_info["cases"]

    cases_pool = []
    if to_gnr_err_type in specific_error_types:
        cases_pool = [to_gnr_err_type]
    else:
        cases_pool = list( set(error_types_all) - set(specific_error_types) )

    msg = []
    temp = {
        "role": "system",
        "content": ""
    }
    msg.append(temp)
    for case in cases:
        if case["in_use"] is False:
            continue
        case_error_type = case["error_types"][0]
        if case_error_type not in cases_pool:
            continue

        if case_error_type not in specific_error_types:
            # not pronoun or negation
            temp_question = sent_prompt.format(case["original_sent"], case["error_types"][0], case["error_types"][0], case["error_types"][0], case["error_types"][0], case["error_types"][0])
            if case["word"] != "no_word":
                temp_response = response_format["word_found"].format(case_error_type, case["word"], case["sent"])
            else:
                temp_response = response_format["no_word_found"].format(case_error_type)
        elif case_error_type == "circumstance" or case_error_type == "complex":
            # pronoun or negation
            temp_question = specific_prompt[case_error_type].format(case["original_sent"])
            if case["word"] != "no_word":
                temp_response = response_format[case_error_type]["word_found"].format(case["sent"])
            else:
                temp_response = response_format[case_error_type]["no_word_found"].format()
        else:
            raise Exception("error type not found")
        
        temp = {
            "role": "user",
            "content": temp_question
        }
        msg.append(temp)
        temp = {
            "role": "assistant",
            "content": temp_response
        }
        msg.append(temp)

    if to_gnr_err_type not in specific_error_types:
        # not pronoun or negation
        temp_question = sent_prompt.format(sent_to_be_uploaded, to_gnr_err_type, to_gnr_err_type, to_gnr_err_type, to_gnr_err_type, to_gnr_err_type)
    elif to_gnr_err_type == "circumstance" or to_gnr_err_type == "complex":
        # pronoun or negation
        temp_question = specific_prompt[to_gnr_err_type].format(sent_to_be_uploaded)
    else:
        raise Exception("error type not found")

    temp = {
        "role": "user",
        "content": temp_question
    }
    msg.append(temp)
    return msg

def generate_sents_file(args, type_list: list, type_pn: str):
    insist_overwrite = False
    loc_info = json_read(loc_info_path)
    output_dir = loc_info["root"] + loc_info["output_dir"]
    output_json_dir = output_dir + "/" + args.save_path
    check_and_mkdir(output_json_dir)

    file_already_exists = True
    for err_type in type_list:
        output_json_path = output_json_dir + "/" + err_type + ".json"
        if not file_exist_check(output_json_path):
            file_already_exists = False
            break
    
    if file_already_exists:
        if not insist_overwrite:
            return
    temp_path = prepare_original_sents(args.num, output_json_path, type_pn)
    for err_type in type_list:
        output_json_path = output_json_dir + "/" + err_type + ".json"
        if output_json_path != temp_path:
            os.system("cp " + temp_path + " " + output_json_path)
    return

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

def extract_from_response(res, err_type, type_pn):
    key_word = {
        "negative": ["pronoun", "Auxiliary"],
        "positive": ["circumstance", "complex"]
    }
    if err_type == key_word[type_pn][0]:
        if key_word[type_pn][0] in res and "not found" in res and "None." in res:
            # "Gender-specific pronoun is not found: None."
            return "no_sent"
        else:
            return res
    elif err_type == key_word[type_pn][1]:
        if key_word[type_pn][1] in res and "not found" in res and "None." in res:
            # "Auxiliary verbs word is not found: None."
            return "no_sent"
        else:
            return res
    else:
        if "^_^" in res:
            return res.split("^_^")[-1]
        else:
            return "no_sent"

def generate_sents_from_file(args, process_id: int, lock, num_process: int, api: str, loc_info_path, type_pn = "negative"):
    loc_info = json_read(loc_info_path)
    output_dir = loc_info["root"] + loc_info["output_dir"]
    # output_json_path = output_dir + "/" + args.save_path + ".json"
    output_json_root = output_dir + "/" + args.save_path
    check_and_mkdir(output_json_root)
    output_json_path = output_json_root + "/" + type_pn + ".json"

    cnn_items = json_read(output_json_path)
    # sents = get_my_sents()

    output_examples = []
    
    prompt_path = loc_info["root"] + loc_info["prompt"]
    prompt_info = json_read(prompt_path)
    response_format = prompt_info["gpt35"][type_pn]["format"]
    cases = prompt_info["gpt35"][type_pn]["cases"]
    error_types = prompt_info["gpt35"][type_pn]["error_types"]
    sent_prompt = prompt_info["gpt35"][type_pn]["sent_prompt"]

    # prompts_check = {}
    with tqdm(total=len(cnn_items)) as pbar:
        pbar.set_description("p_id: " + str(process_id) + " - generate sents")
        index = -1
        for cnn_item in cnn_items:
            index += 1
            if index % num_process != process_id:
                pbar.update(1)
                continue
            if cnn_item["status"] == "finished":
                pbar.update(1)
                continue
            cnn_item["status"] = "in_progress"
            for err_type in error_types:
                if cnn_item["result"][err_type]["finished"]:
                    continue
                pbar.set_description("p_id: " + str(process_id) + " err_type: " + err_type)
                # sent = cnn_item["claim_sent"]
                all_claim = cnn_item["claim"]
                sent = all_claim
                time.sleep(0.2)
                try:
                    with timeout(openai_threshold, exception=RuntimeError):
                        msg_path = ["msg"]
                        # messages = get_input_messages(*msg_path)
                        messages = get_input_msg(sent, err_type, type_pn)

                        # lock.acquire()
                        # path = os.getcwd() + "/temp_data/prompt_check.json"
                        # prompts_check = json_read(path)
                        # prompts_check[err_type] = messages
                        # json_write(path, prompts_check)
                        # lock.release()

                        T1 = time.time()
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
                        # response = "aaa^_^bbb"

                        T2 = time.time()
                        res = response['choices'][0]['message']['content']

                        cnn_item["result"][err_type]["finished"] = True
                        cnn_item["result"][err_type]["response"] = res
                        cnn_item["result"][err_type]["pert_sent"] = extract_from_response(res, err_type, type_pn)
                        cnn_item["result"][err_type]["elapsed_time"] = T2 - T1

                        lock.acquire()
                        output = json_read(output_json_path)
                        for type_ in error_types:
                            output[index]["result"][type_] = cnn_item["result"][type_]
                            output[index]["status"] = cnn_item["status"]
                        json_write(output_json_path, output) # cnn_items
                        cnn_item = output[index]
                        lock.release()
                except RuntimeError:
                    print_hint_with_color("timeout", "red", " generating sents")
                except openai.error.RateLimitError as e:
                    if e._message.startswith("You exceeded your current quota,"):
                        # api_set.remove(openai.api_key)
                        print_hint_with_color("Switching to API key: ", "red", openai.api_key)
                    else:
                        print_hint_with_color("error_msg ", "red", e._message[:60])
                    time.sleep(20)
                    error_flag = True

            lock.acquire()
            output = json_read(output_json_path)
            finish_status = True
            for key in output[index]["result"]:
                if output[index]["result"][key]["finished"] is False:
                    finish_status = False
                    break
            if finish_status:
                output[index]["status"] = "finished"
            else:
                output[index]["status"] = "unfinished"
            cnn_item = output[index]
            json_write(output_json_path, output) # cnn_items

            if index % 2 == 0 and cnn_item["status"] == "finished":
                backup_dir = "/".join(output_json_path.split("/")[:-1]) + "/backup/"
                check_and_mkdir(backup_dir)
                temp = backup_dir + output_json_path.split("/")[-1]
                save_path = temp[:-5] + "_" + str(index) + ".json"
                os.system("cp " + output_json_path + " " + save_path)
            lock.release()

            pbar.update(1)

    # json_write(output_json_path, cnn_items)
    # xls_write(output_json_path)
    
    print_hint_with_color("   [[Job done]] ", "green", "Results are in: " + output_json_path)

def loc_json_modify():
    root_dir = os.getcwd()
    loc_info_path = root_dir + "/loc_info.json"
    loc_info = json_read(loc_info_path)
    loc_info["root"] = root_dir + "/"
    json_write(loc_info_path, loc_info)
    return loc_info_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--num", default=6, type=int)
    parser.add_argument("--save_path", default="positive", type=str)
    args = parser.parse_args()

    loc_info_path = loc_json_modify()
    loc_info = json_read(loc_info_path)

    # type_list = ["negative", "positive_1", "positive_2", "positive_3"]
    type_list = ["positive"]
    if True:
        for type_pn in type_list:
            generate_sents_file(args, type_list, type_pn)

    if True:
        api_list = [
            # "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",
            "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" # 120usd
        ]
        type_pn = type_list[0]
        num_process = len(api_list)
        lock = Lock()
        process_list = []
        for p_id in range(num_process):
        # for type_pn in type_list:
            p = Process(target=generate_sents_from_file, args=(args, p_id, lock, num_process, api_list[0], loc_info_path, type_pn))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
    print_hint_with_color("end", "green")
