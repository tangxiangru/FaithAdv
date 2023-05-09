import json
from typing import List
loc_info_path = "../loc_info.json"

def json_read(filePath):
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data

def json_write(filePath, data):
    """
    write a list of string to a file in easy-read way
    """
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=4)

def file_exist_ensure(filePath):
    """
    ensure the file exists
    if not, create the file and return False
    if yes, return True
    """
    import os
    if not os.path.exists(filePath):
        os.makedirs(filePath)
        return False
    else:
        return True

def get_prompt(module_name: str) -> str:
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
    sent = ""
    subject_prompt += "original_sentence: " + sent
    return subject_prompt

def get_contents_in_breaces(text: str) -> list:
    """
    get the contents in the braces
    """
    import re
    contents = re.findall(r'\[(.*?)\]', text)
    return contents

def get_grammar_content_from_json(json_path: str) -> List[str]:
    """
    get the grammar content from the json file
    """
    json_list = json_read(json_path)
    strucs = []
    for item in json_list:
        strucs.append(item["perturbed_sent"]["structure"])
    grammar_content = []
    for struc in strucs:
        grammar_content += get_contents_in_breaces(struc)
    grammar_content = list(set(grammar_content))
    grammar_content.sort()
    return grammar_content

if __name__ == "__main__":
    res_json_path = "/Users/gmh/Library/CloudStorage/OneDrive-zju.edu.cn/code/python_vscode/NLP_zly/ROSE_NLI/output/result.json"
    grammars = get_grammar_content_from_json(res_json_path)
    for grammar in grammars:
        print(grammar)

