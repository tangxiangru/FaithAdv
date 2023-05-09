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
        model=args.model,
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
            model=args.model,
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
            print("same sentence. {} chances left to try.".format(try_times))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="text-curie-001", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--input_prompt", required=True, type=str)
    parser.add_argument("--input_sentence", default="./json_file/sents0.json", type=str)
    args = parser.parse_args()

    # ########################## params to adjust manually ##########################
    quantity_of_original_sents_to_generate = 100
    replica_of_each_original_sent = 1
    output_dir = "outputs/result"
    csv_sep_read = "ᚢ"
    csv_sep_write = "ᚢ"     # "#####"
    word_similarity_threshold = 0.6
    debugging = False
    json_process = True
    # ########################## params to adjust manually ##########################

    os.makedirs(output_dir, exist_ok=True)

    api_set = set(api_list)
    openai.api_key = random.sample(list(api_set), 1)[0]
    
    prompt_string = prepare_prompt(args.input_prompt, sep=csv_sep_read)
    prompt_string_original = prepare_prompt(args.input_prompt, sep=csv_sep_read)
    file_input_sentence = args.input_sentence
    if json_process:
        sents = json_read(file_input_sentence)
    else:
        sents = prepare_original_sents(quantity_of_original_sents_to_generate)

    output_examples = []
    
    cur_id = 0
    deleted = 0
    with tqdm(total=len(sents)) as pbar:
        while cur_id - deleted < len(sents):
            if json_process:
                if sents[cur_id - deleted]["perturbed_sentence"] != "":
                    cur_id += 1
                    pbar.update(1)
                    continue
                if cur_id % 3 == 0:
                    json_write(file_input_sentence, sents)

            # time.sleep(5)
            openai.api_key = random.sample(list(api_set), 1)[0]
            try:
                sent = sents[cur_id - deleted]["sentence"] if json_process else sents[cur_id - deleted]
                sent = sent.strip()

                prompt_string += f'Original sentence: \n{sent}\nPerturbed sentence: \n'
                prompt_string_temp = prompt_string_original + f'Original sentence: \n{sent}\nPerturbed sentence: \n'

                # noSentGenerated = False
                for i in range(replica_of_each_original_sent):
                    sent, perturbed_sent, gpt3_result = generate_perturbed_sent(sent, prompt_string_temp, word_similarity_threshold, debugging)
                    if sent == "noSentGenerated":
                        print("no sent generated, delete this pair.")
                        del sents[cur_id - deleted]
                        deleted += 1
                        json_write(file_input_sentence, sents)
                        # noSentGenerated = True
                        continue
                    if json_process:
                        sents[cur_id - deleted]["perturbed_sentence"] = perturbed_sent
                    else:
                        if debugging:
                            tmp = {
                                "original_sent": sent,
                                "perturbed_sent": perturbed_sent,
                                "gpt3_judge_if_same": gpt3_result
                            }
                        else:
                            tmp = {
                                "original_sent": sent,
                                "perturbed_sent": perturbed_sent
                            }
                        output_examples.append(tmp)
                # if noSentGenerated:
                #     continue
                cur_id += 1
                pbar.update(1)
                    
            except openai.error.RateLimitError as e:
                if e._message.startswith("You exceeded your current quota,"):
                    api_set.remove(openai.api_key)
                    print("Switching to API key: ", openai.api_key)
                else:
                    print(e._message[:20])
                time.sleep(20)
                error_flag = True
    if json_process:
        json_write(file_input_sentence, sents)
        print()
        print(">>> {} suceeded, {} failed.".format(len(sents), deleted))
        print(">>> final results have been put into {}, please check.".format(file_input_sentence))
        input("    ############################ Press Enter To Exit ###########################")
        print()
    else:
        output_file = args.input_prompt.split('/')[-1].split('.')[0] + ".csv"
        output_prompt_file = args.input_prompt.split('/')[-1].split('.')[0] + ".txt"
        output_file = os.path.join(output_dir, output_file)
        output_prompt_file = os.path.join(output_dir, output_prompt_file)

        df = pd.DataFrame(output_examples)
        # df.to_csv(output_file, index=False, sep=csv_sep_write, encoding="utf-8") # not support multi-character separator
        np.savetxt(output_file, df, delimiter=csv_sep_write, header=csv_sep_write.join(df.columns.values), fmt='%s', comments='', encoding=None) # this way support multi-character separator
        with open(output_prompt_file, "w") as f:
            f.write(prompt_string)

        print(f"API have not been used {api_set}")
