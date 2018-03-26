import re
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import data_helper
import random
import time
from tensorflow.contrib import learn
from konlpy.tag import Mecab
from konlpy.utils import pprint
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

logging.getLogger().setLevel(logging.INFO)

def preprocess(text):
    preprocess.stopwordList = getattr(preprocess, 'stopwordList', None)

    target_list = ["\t", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", ":phone:", "☏", "※", ":arrow_forward:", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴", "┘"]

    if not preprocess.stopwordList:
        print('cnn_run.py - read stopword...')
        fp = open(dir_path + '/../textrank/config/stopword.txt', 'r')

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

        if not preprocess.stopwordList:
            stopwordList = []
            stopwordList = set(map(lambda x: x.strip(), fp.readlines()))
            preprocess.stopwordList = stopwordList
            fp.close()
        textList = list(map(lambda x: x + '. ', list(filter(lambda x: x in preprocess.stopwordList, textList))))
        
        textList = (''.join(textList)).strip()
        textList = textList.replace('.', '. ')
    else:
        textList = ""

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

    if len(result) > 300:
        result = result[0:300]
    counter_konlpy += 1
    #sys.stdout.write("\rParsed: %d / %d" %(counter_konlpy, total_dataset))
    #sys.stdout.flush()
    return ' '.join(result)

def get_x_test(contents):
    """Step 1: load data for prediction"""
    columns = ['section', 'class', 'subclass', 'abstract']
    selected = ['section', 'abstract']
    #test_list = test_list[0:10000]
    data = []
    #print("Listing all datas in testset.")
    start = time.time()
    for content in contents:
        #print(content)
        data.append(['', '', '', content])
    df = pd.DataFrame(data, columns=columns)
    global counter_konlpy
    global total_dataset
    start = time.time()
    counter_konlpy = 0
    total_dataset = len(contents)
    #x_raw = [example['abstract'] for example in test_examples]
    #x_test = [data_helper.clean_str(x) for x in x_raw]
    x_test = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    #print("\nExecution time = {0:.5f}".format(time.time() - start))

    logging.info('The number of x_test: {}'.format(len(x_test)))


    return x_test

def get_classification_models():
    checkpoint_dir =  dir_path + '/trained_model/'
    models = []
    try:
        dirs = list(os.walk(checkpoint_dir))[0][1]
        models = dirs
    except:
        models = []

    return models

def get_category_name(model_name):
    get_category_name.model_names = getattr(get_category_name, 'model_names', {})

    try:
        if model_name not in get_category_name.model_names:
            print("load %s" % (model_name))
            checkpoint_dir =  dir_path + '/trained_model/' + str(model_name)
            category_name = ''
            f = open(checkpoint_dir + '/category.json')
            category_name = json.loads(''.join(f.readlines()))['name']
            get_category_name.model_names[model_name] = category_name
            f.close()
    except Exception as e:
        pass

    return get_category_name.model_names[model_name]

def predict_unseen_data(x_test, model_name):
    """Step 0: load trained model and parameters"""
    predict_unseen_data.model_names = getattr(predict_unseen_data, 'model_names', {})

    if model_name not in predict_unseen_data.model_names:
        print("load checkpoint, labels, vocab_processor in %s" % (model_name))
        checkpoint_dir =  dir_path + '/trained_model/' + str(model_name)
        if not checkpoint_dir.endswith('/'):
            checkpoint_dir += '/'
        f = open(checkpoint_dir + 'labels.json')
        label_data = ''.join(f.readlines())
        labels = json.loads(label_data)
        f.close()
        f = open(dir_path + '/parameters.json')
        params = json.loads(f.read())
        f.close()
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
        logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

        vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        predict_unseen_data.model_names[model_name] = { }
        predict_unseen_data.model_names[model_name]['checkpoint_file'] = checkpoint_file
        predict_unseen_data.model_names[model_name]['vocab_processor'] = vocab_processor
        predict_unseen_data.model_names[model_name]['parameters'] = params
        predict_unseen_data.model_names[model_name]['labels'] = labels

    checkpoint_file = predict_unseen_data.model_names[model_name]['checkpoint_file']
    vocab_processor = predict_unseen_data.model_names[model_name]['vocab_processor']
    labels = predict_unseen_data.model_names[model_name]['labels']
    params = predict_unseen_data.model_names[model_name]['parameters']
    

    x_test = np.array(list(vocab_processor.transform(x_test)))

    
    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 0})
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    all_predictions = list(map(lambda x: int(np.asscalar(x)), all_predictions))

    all_predictions = list(map(lambda x: labels[x], all_predictions))
    return all_predictions
