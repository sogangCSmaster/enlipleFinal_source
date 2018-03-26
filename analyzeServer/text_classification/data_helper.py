import re
import time
import logging
import numpy as np
import pandas as pd
import random
import json
import sys
import os
from collections import Counter
from konlpy.tag import Mecab
from konlpy.utils import pprint

def preprocess(text):

    target_list = ["\t", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", ":phone:", "☏", "※", ":arrow_forward:", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴", "┘"]

    fp = open('./stopwords.txt', 'r')

    for target in target_list:
        text = text.replace(target, " ")    
    regularExpression1 = "\r?\n|\r|\t"
    regularExpression2 = "[a-z0-9_+]+@([a-z0-9-]+\\.)+[a-z0-9]{2,4}|[a-z0-9_+]+@([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
    regularExpression3 = "(file|gopher|news|nntp|telnet|https?|ftps?|sftp):\\/\\/([a-z0-9-]+\\.)+[a-z0-9]{2,4}|(file|gopher|news|nntp|telnet|h    ttps?|ftps?|sftp):\\/\\/([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
    regularExpression4 = "([a-z0-9-]+\\.)+[a-z0-9]{2,4}|([a-z0-9-]+\\.)+([a-z0-9-]+\\.)+[a-z0-9]{2,4}"
    regularExpression5 = "\\(.*?\\)|\\[.*?\\]|【.*?】|<.*?>"
    regularExpression6 = "[!@+=%^;:]"
    regularExpression7 = "[ ]{1,20}"
    regularExpression8 = "[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z]기자|[가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z]기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z] 기자|[가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z][가-힣a-zA-Z]기자" 
    
    part1 = re.compile(regularExpression1)
    part2 = re.compile(regularExpression2)
    part3 = re.compile(regularExpression3)
    part4 = re.compile(regularExpression4)
    part5 = re.compile(regularExpression5)
    part6 = re.compile(regularExpression6)
    part7 = re.compile(regularExpression7)
    part8 = re.compile(regularExpression8)
    text = re.sub(part1, "", text)
    text = re.sub(part2, "", text)
    text = re.sub(part3, "", text)
    text = re.sub(part4, "", text)
    text = re.sub(part5, "", text)
    text = re.sub(part6, " ", text)
    text = re.sub(part7, " ", text)
    text = re.sub(part8, "", text)
    

    trimPoint = text.rfind('다.')
    if trimPoint > -1:
        try:
            text = text[0:trimPoint+2]
        except Exception as e:
            print(e)
    if text:
        textList = re.split('\\. |\\.', text)
        textList.pop()

        stopwordList = []
        stopwordList = set(map(lambda x: x.strip(), fp.readlines()))
        textList = list(map(lambda x: x + '. ', list(filter(lambda x: x in stopwordList, textList))))
        
        textList = (''.join(textList)).strip()
        textList = textList.replace('.', '. ')
    else:
        textList = ""

    fp.close()
    return text

def clean_str(s):
    """Clean sentence"""
    global counter_konlpy
    global total_dataset
    #global stopwords
    s = re.sub('[0-9]', '', s)
    s = preprocess(s)

    mecab = Mecab()
    #print(' '.join(kkma.nouns(s)))
    result = []
    result = mecab.nouns(s)
    #temp = []
    #temp = mecab.nouns(s)
    #for noun in temp:
        #flag = 0;
        #for sword in stopwords:
            #if noun == sword:
                #flag = 1;
                #break;
        #if flag == 0:
            #result.append(noun)     

    if len(result) > 1000:
        result = result[0:1000]
    counter_konlpy += 1
    #sys.stdout.write("\rParsed: %d / %d" %(counter_konlpy, total_dataset))
    #sys.stdout.flush()
    return ' '.join(result)


def load_data_and_labels(foldername):
    """Load sentences and labels"""
    columns = ['section', 'class', 'subclass', 'abstract']
    selected = ['section', 'abstract']
    global counter_konlpy
    global total_dataset
    #global stopwords
    #stopword_file = "./stopwords.json"
    #stopwords = tuple(json.loads(open(stopword_file).read()))
    file_list = []
    for path, dirs, files in os.walk(foldername):
        if files:
            for filename in files:
                fullname = os.path.join(path, filename)
                file_list.append(fullname)
    random.shuffle(file_list)


    data = []
    print("Listing all datas in dataset.")
    start = time.time()
    for filename in file_list:
        fp = open(filename, 'r', encoding='utf-8')
        temp = fp.readlines()
        data.append([filename.split('/')[2], filename.split('/')[3], filename.split('/')[4], ''.join(temp)])
        fp.close()
    df = pd.DataFrame(data, columns=columns)
    print("Execution time = {0:.5f}".format(time.time() - start))

    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    print("Parsing dataset with Konlpy.")
    start = time.time()
    counter_konlpy = 0
    total_dataset = len(file_list)
    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    print("\nExecution time = {0:.5f}".format(time.time() - start))
    return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    dataset = './dataset'
    #dataset = './dataset/description'
    load_data_and_labels(dataset)
