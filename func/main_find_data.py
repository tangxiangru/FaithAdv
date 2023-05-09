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
import xlwt
from func.temp import *

loc_info_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/loc_info.json"
# loc_info_path = "./loc_info.json"
api_list = [
    # "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",
    # "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",
    # "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",
    # above keys are unavailable
    "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",
    "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",
    "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",
    # below keys will be used in the future
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    # "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",
    ]

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
    file_exist_check(filePath)
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=4)

def file_exist_check(filePath: str):
    """
    check if a file exists
    if not, create it, and return False
    if yes, return True
    """
    if not os.path.exists(filePath):
        open(filePath, 'w').close()
        return False
    else:
        return True

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

def get_prompt(module_name: str, sent: str) -> str:
    """
    get the prompt of the task
    """
    loc_info = json_read(loc_info_path)
    prompt_path = loc_info["root"] + loc_info["prompt"]
    prompt_info = json_read(prompt_path)
    
    subject_info = prompt_info[module_name]
    subject_prompt = subject_info["prompt"] + "\n"
    for item in subject_info["cases"]:
        subject_prompt += "original_sentence: " + item["original_"] + "\n" + "perturbed_sentence: " + item["perturbed"] + "\n\n"
    subject_prompt += "original_sentence: " + sent + "\n" + "perturbed_sentence: "
    return subject_prompt

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

def remove_return(sent: str):
    """
    remove the new line character '\n' in a string
    """
    return sent.replace("\n", " ")

def prepare_original_sents(quantity: int) -> List[str]:
    # quantity is the number of sentences to generate
    sents = generate_sentences(quantity, datasetName = "cnn_dailymail", version = "3.0.0", split = "train")
    return sents

def prepare_prompt(input_prompt_file) -> str:
    if input_prompt_file.endswith(".txt"):
        prompt_text = open(args.input_prompt).read()
    elif input_prompt_file.endswith(".csv"):
        df = pd.read_csv(input_prompt_file)
        original_sents = df["original_sent"].tolist()
        perturbed_sents = df["perturbed_sent"].tolist()
        prompt_text = "Change the meaning of the sentence by perturbing the sentence in word-level and ensure that the change is no more than 3 words.\n\n"
        for i in range(len(original_sents)):
            prompt_text += f'Original sentence: \n{original_sents[i]}\nPerturbed sentence: \n{perturbed_sents[i]}\n\n'
    return prompt_text.strip() + "\n\n"

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
    assis_types = list(msg_dict["cases"][-1]["assistant"].keys())
    if "structure" in assis_types:
        assis_types.remove("structure")
    return assis_types    

def extract_perturbed_sent(response, msg_path):
    """
    extract the perturbed sentence from the response
    """
    # response = response.strip()

    # find the [x] in the response
    indexes = []
    content = []
    len_of_index = 0
    while True:
        index = response.find("[{}]".format(len_of_index))
        if index != -1:
            indexes.append(index)
            len_of_index += 1
        else:
            break
    for i in range(len(indexes)):
        char_offset = 3 if i < 10 else 4
        if i == len(indexes) - 1:
            content.append(remove_return(response[indexes[i]+char_offset : ]))
        else:
            content.append(remove_return(response[indexes[i]+char_offset : indexes[i+1]]))
    # todo check if they are same
    # for sent in content:
    #     if gpt35_same_meaning(content[0], sent) == "yes":
    #         return sent
    try:
        assis_types = get_types(*msg_path)
        type_num = len(assis_types)
    
        content_dict = {
            "response": response
            # "structure": content[0],
            # "subject": content[1],
            # "verb": content[2],
            # "object": content[3]
        }
        for i in range(type_num):
            content_dict[assis_types[i]] = content[i+1]
    except:
        content_dict = None
    return content_dict

def add_quota(sent: str) -> str:
    """
    add quota to a string
    """
    return f'"{sent}"'

def get_input_messages(*args) -> list:
    loc_info = json_read(loc_info_path)
    prompt_path = loc_info["root"] + loc_info["prompt"]
    msg_dict = json_read(prompt_path)
    # find the layer in the json hierarchy
    for arg in args:
        msg_dict = msg_dict[arg]
    # # get the len of the assistant
    # assis_types = list(msg_dict["cases"][0]["assistant"].keys())
    # if "structure" in assis_types:
    #     assis_types.remove("structure")

    args = list(args)
    assis_types = get_types(*args)
    type_num = len(assis_types)
    # get the system message content
    system_content = msg_dict["system"]["requirement"]
    for i in range(type_num):
        system_content += msg_dict["system"]["iterate_way"].format(i+1, assis_types[i], assis_types[i], assis_types[i])
    system_content += msg_dict["system"]["ret_format"]
    for i in range(type_num):
        system_content += msg_dict["system"]["iterate_ret_format"].format(i+1, assis_types[i])

    msg = []
    temp = {
        "role": "system",
        "content": system_content
    }
    msg.append(temp)
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
            num += 1
            sents.append(item["assistant"][key])
        # content = "[0]{}\n[1]{}\n[2]{}\n".format(sents[0], sents[1], sents[2])
        assistant_content = ""
        for i in range(num):
            assistant_content += f"[{i}]{sents[i]}\n"
        temp = {
            "role": "assistant",
            "content": assistant_content
        }
        msg.append(temp)
    return msg

def json_print_to_file(json_path: str):
    txt_path = json_path.replace(".json", ".txt")
    content = json_read(json_path)
    with open(txt_path, "w") as f:
        f.write(json.dumps(content, indent=4))
    return txt_path

def xls_write(json_path):
    xls_path = json_path.replace(".json", ".xls")
    content = json_read(json_path)
    file_exist_check(xls_path)

    workbook = xlwt.Workbook(encoding= 'ascii')
    worksheet = workbook.add_sheet("result")

    row_base = 1
    row_interval = 18
    col_base = 1
    i = 0
    for item in content:
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

def same_meaning_gpt35(sent1: str, sent2: str) -> bool:
    """
    check if two sentences have the same meaning
    """
    messages=[
        {"role": "system", "content": "Determine whether the meaning of the two sentences is similar. If they are similar, answer \"yes\", otherwise, answer \"no\"."},
        {"role": "user", "content": "{}\n{}".format(sent1, sent2)}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messages,
        temperature=0.7,
        # max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    res = response['choices'][0]['message']['content']
    if "yes" in res.lower():
        return True
    elif "no" in res.lower():
        return False
    else:
        return False

def similarity_check(sent: str, sents: dict) -> dict:
    """
    check if the sent is similar to any sent in sents
    """
    for s_key in sents:
        if s_key == "response" or s_key == "structure":
            continue
        if same_meaning_gpt35(sent, sents[s_key]):
            sents[s_key] = "not_valid"
    return sents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="text-davinci-003", type=str)
    # parser.add_argument("--input_prompt", required=True, type=str)
    parser.add_argument("--num", default=2, type=int)
    parser.add_argument("--type", default="result", type=str)

    args = parser.parse_args()

    api_set = set(api_list)
    openai.api_key = random.sample(list(api_set), 1)[0]
    
    # prompt_string = prepare_prompt(args.input_prompt)
    sents = prepare_original_sents(args.num)
    # sents.append("The 31-year-old could be sold by City as they look to reshape their squad. ")
    output_examples = []
    
    loc_info = json_read(loc_info_path)
    output_dir = loc_info["root"] + loc_info["output_dir"]
    output_json_path = output_dir + "/" + args.type + ".json"
    cur_id = 0
    with tqdm(total=len(sents)) as pbar:
        while cur_id < len(sents):
            time.sleep(0.3)
            openai.api_key = random.sample(list(api_set), 1)[0]
            try:
                sent = sents[cur_id]

                # prompt_string = get_prompt(args.type, sent)
                # f'Original sentence: \n{sent}\nPerturbed sentence: \n'
                
                msg_path = ["msg"]
                messages = get_input_messages(*msg_path)
                messages.append({
                    "role": "user",
                    "content": sent
                })
                # print_hint_with_color("messages", "cyan", messages)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = messages,
                    temperature=0.7,
                    # max_tokens=64,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                res = response['choices'][0]['message']['content']
                perturbed_sent = extract_perturbed_sent(res, msg_path)
                if perturbed_sent != None:
                    perturbed_sent = similarity_check(sent, perturbed_sent)
                    tmp = {
                        "original_sent_": sent,
                        "perturbed_sent": perturbed_sent
                        # "gpt3_says_they_have_same_meaning": is_same
                    }
                    output_examples.append(tmp)
                cur_id += 1
                pbar.update(1)
                json_write(output_json_path, output_examples)

            except openai.error.RateLimitError as e:
                if e._message.startswith("You exceeded your current quota,"):
                    api_set.remove(openai.api_key)
                    print("Switching to API key: ", openai.api_key)
                else:
                    print(e._message[:20])
                time.sleep(20)
                error_flag = True

    json_write(output_json_path, output_examples)
    xls_write(output_json_path)
    
    print_hint_with_color("   [[Job done]] ", "green", "Results are in: " + output_json_path)
