import pandas as pd
import numpy as np
import os

def write_list_to_file(file_name, list_of_str):
    """
    write a list of string to a file in easy-read way
    """
    with open(file_name, 'w') as f:
        f.write("sents\n")
        for item in list_of_str:
            f.write("%s \n" % item)

def remove_return(sent: str):
    """
    remove the new line character \n in a string
    """
    return sent.replace("\n", " ")

def csv2easyReadTxt(filePath, break_line: str, csv_sep: str):
    """
    the csv file should have two columns: original_sent and perturbed_sent.\n
    read a csv file and write the content to a file in easy-read way.\n
    the output is in the same directory.
    """
    cents_orig = pd.read_csv(filePath, sep=csv_sep, engine='python', error_bad_lines=False)
    print(cents_orig.shape)
    print(cents_orig.__class__)
    cents = []
    if cents_orig.shape[1] == 2:
        for i in range(cents_orig.shape[0]):
            cents.append(remove_return(cents_orig['original_sent'][i]))
            cents.append(remove_return(cents_orig['perturbed_sent'][i]))
            cents.append(break_line)
    else:
        for i in range(cents_orig.shape[0]):
            cents.append(remove_return(cents_orig['original_sent'][i]))
            cents.append(remove_return(cents_orig['perturbed_sent'][i]))
            cents.append(break_line + " gpt3_judge_if_same: " + remove_return(cents_orig['gpt3_judge_if_same'][i]))
    file_name = filePath + ".easyread.csv"
    write_list_to_file(file_name, cents)

def write_csv_to_file(file_name, df, csv_sep: str):
    # df.to_csv(file_name, index=False, sep=csv_sep)
    np.savetxt(file_name, df, delimiter=csv_sep, header=csv_sep.join(df.columns.values), fmt='%s', comments='', encoding=None)

def easyReadTxt2csv(filePath, csv_sep: str):
    cents = pd.read_csv(filePath, sep='\t')
    print(cents.shape)
    print(cents.__class__)
    len0 = cents.shape[0]
    len = int(cents.shape[0]/3)
    index1 = range(0, cents.shape[0], 3)
    index2 = range(1, cents.shape[0], 3)
    index_df = range(0, int(cents.shape[0]/3))
    original_sent = cents['sents'][index1]
    perturbed_sent = cents['sents'][index2]
    df_list = []
    for i, j in zip(original_sent, perturbed_sent):
        temp = {
            'original_sent': i,
            'perturbed_sent': j
        }
        df_list.append(temp)
    file_name = filePath + ".standard.csv"
    new_df = pd.DataFrame(df_list)
    write_csv_to_file(file_name, new_df, csv_sep)

def changeCSVsep(filePath, csv_sep_read, csv_sep_write):
    break_line = "# ----------------------------------------"
    csv2easyReadTxt(filePath, break_line, csv_sep_read)
    easyReadTxt2csv(filePath + ".easyread.csv", csv_sep_write)

    path_to_delete = [filePath, filePath + ".easyread.csv"]
    for path in path_to_delete:
        if os.path.exists(path):
            os.remove(path)
        else:
            print("no such file: ", path)
    os.rename(filePath + ".easyread.csv" + ".standard.csv", filePath)

def func1():
    # ######################### Easy Read #########################
    filePath = "adversarial_data_generation/prompts/prompt_own.csv"
    filePath = "outputs/result_temp/result12/prompt_own.csv"
    # filePath = "outputs/result_temp/result12/prompt_own copy 6.csv"
    break_line = "# ----------------------------------------"
    csv_sep_read = "ᚢ"
    csv2easyReadTxt(filePath, break_line, csv_sep_read)

def func2():
    # ######################### Standard CSV #########################
    filePath = "adversarial_data_generation/prompts/prompt_own_pool.csv"
    csv_sep_write = "ᚢ"
    easyReadTxt2csv(filePath, csv_sep_write)

def func3():
    # ######################### change CSV sep #########################
    filePath = "adversarial_data_generation/prompts/test.csv"
    csv_sep_read = "#$$$#^^^#" # "ᚢ"
    csv_sep_write = "ᚢ"
    changeCSVsep(filePath, csv_sep_read, csv_sep_write)

if __name__ == "__main__":
    func1()
    # func2()
    # func3()
    pass
