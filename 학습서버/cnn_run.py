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

def clean_str(s):
    """Clean sentence"""
    global counter_konlpy
    global total_dataset
    s = re.sub('[0-9]', '', s)
    mecab = Mecab()
    result = []
    result = mecab.nouns(s)
    if len(result) > 300:
        result = result[0:300]
    counter_konlpy += 1
    sys.stdout.write("\r Parsed: %d / %d" %(counter_konlpy, total_dataset))
    sys.stdout.flush()
    return ' '.join(result)

def predict_unseen_data(contents):
    """Step 0: load trained model and parameters"""
    params = json.loads(open(dir_path + '/parameters.json').read())
    checkpoint_dir =  dir_path + '/trained_model'
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

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
    #print("Execution time = {0:.5f}".format(time.time() - start))

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = json.loads(open(dir_path + '/labels.json').read())
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
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

    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
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

    return all_predictions
