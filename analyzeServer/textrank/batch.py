from textrank import TextRank
import time
import os
import signal
import logging
import sys
import numpy as np
import traceback
from multiprocessing import Process, Queue, Value
from datetime import datetime
import server
import socket
from konlpy.tag import Mecab
from contextlib import closing
from urllib.parse import urlparse

sys.path.append('../text_classification')
from cnn_run import predict_unseen_data, get_x_test, get_classification_models, get_category_name
sys.path.append('./config')
from pyredis import Redis
from db import DB
from mongo import Mongo
from config import load_config
import re
cwd = os.getcwd()


# logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)




def check_socket(host, port):
    # server check
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(2)
        if sock.connect_ex((host, port)) == 0:
            return True
        else:
            return False

# config/singleword.txt
def get_singlewords():
    global global_singlewords
    return global_singlewords

# config/stopword.txt
def get_stopwords(redis, root_domain):
    global global_stopwords
    return global_stopwords.union(redis.get_stopwords(root_domain))

def ml_classifications(db, mongo, data, x_test, classification_models, bulk_op):
    start_time = time.time()
    logging.debug('*********************************')
    logging.debug('ml classification start time : %f' %(start_time))
    # initialize variable
    classifications = { }

    
    for classification_model in classification_models:
        model_name = classification_model
        category_name = get_category_name(model_name)

        if model_name is None:
            # model does not exist
            logging.debug('model not found')
            continue

        # get category options
        category_options = predict_unseen_data(x_test, model_name)
        # convert np array to python list

        # create classifications object
        if model_name == 'trained_model_sentiment':
          for i in range(len(category_options)):
              URI = data[i][0]
              mongo.bulk_insert_sentiment(bulk_op, URI, category_options[i])
        else:
          for i in range(len(category_options)):
              URI = data[i][0]

              # initialize empty key
              if not classifications.get(URI):
                  classifications[URI] = []
              classifications[URI].append({
                  'name': category_name,
                  'value': category_options[i]
              })

    mongo.bulk_insert_ml_classifications(bulk_op, classifications)
    end_time = time.time()
    logging.debug("ml classification end time : %f" %(end_time))
    logging.debug("ml classification total execution time : %f" %(end_time - start_time))
    logging.debug("**************************************")


# 키워드를 추출하고 mongo에 bulk insert하는 함수
def keyword(mongo, redis, tagger, data, bulk_op):
    start_time = time.time()
    logging.debug("\n**********************************")
    logging.debug("keyword extraction start time : %f" %(start_time))

    singlewords = get_singlewords()
    coef = load_config()['coef']
    nnp_addition_multiplier = load_config()['nnp_addition_multiplier']
    nng_addition_multiplier = load_config()['nng_addition_multiplier']
    title_word_addition_multiplier = load_config()['title_word_addition_multiplier']
    minimum_low_freq = load_config()['minimum_low_freq']
    low_freq_word_subtraction_multiplier = load_config()['low_freq_word_subtraction_multiplier']

    for idx, (URI, title, content, root_domain, wordcount) in enumerate(data):
        # get stopwords from redis
        stopwords = get_stopwords(redis, root_domain)
        tr = TextRank(tagger=tagger, window=5, content=content, stopwords=stopwords, singlewords=singlewords, title=title, coef=coef, title_word_addition_multiplier=title_word_addition_multiplier, minimum_low_freq=minimum_low_freq, low_freq_word_subtraction_multiplier=low_freq_word_subtraction_multiplier, nnp_addition_multiplier=nnp_addition_multiplier, nng_addition_multiplier=nng_addition_multiplier)

        # build keyword graph
        tr.keyword_rank()

        # get keyword 키워드의 개수는 최대 15개로 제한
        keywords = tr.keywords(num=15)
        sys.stdout.write("\rkeyword extracted: %d / %d" %(idx, len(data)))
        mongo.bulk_insert_keywords(bulk_op, URI, keywords)

    end_time = time.time()
    logging.debug("keyword extraction end time : %f" %(end_time))
    logging.debug("keywords total execution time : %f" %(end_time - start_time))
    logging.debug("*************************************\n")

# 문장을 추출하고 mongo에 bulk insert하는 함수
def sentence(mongo, redis, tagger, data, bulk_op):
    start_time = time.time()
    logging.debug("\n************************************")
    logging.debug("sentence process start time : %f" %(start_time))

    singlewords = get_singlewords()
    coef = load_config()['coef']
    title_word_addition_multiplier = load_config()['title_word_addition_multiplier']
    minimum_low_freq = load_config()['minimum_low_freq']
    low_freq_word_subtraction_multiplier = load_config()['low_freq_word_subtraction_multiplier']
    # get keywords, sentences using textrank algorithm
    for idx, (URI, title, content, root_domain, wordcount) in enumerate(data):
        # get stopwords from redis
        stopwords = get_stopwords(redis, root_domain)
        tr = TextRank(tagger=tagger, window=5, content=content, stopwords=stopwords, singlewords=singlewords, title=title, coef=coef, title_word_addition_multiplier=title_word_addition_multiplier, minimum_low_freq=minimum_low_freq, low_freq_word_subtraction_multiplier=low_freq_word_subtraction_multiplier)

        # build sentence graph
        tr.sentence_rank()
    
        # wordcount의 개수에 따라 요약율 변경
        summarize_rate = 0.3
        if wordcount < 500:
            summarize_rate = 0.3
        elif wordcount <= 1000:
            summarize_rate = 0.3
        elif wordcount <= 2000:
            summarize_rate = 0.2
        elif wordcount <= 3000:
            summarize_rate = 0.1

        # get sentence
        sentences = tr.sentences(summarize_rate)
        sys.stdout.write("\rsentence extracted: %d / %d" %(idx, len(data)))
        mongo.bulk_insert_sentences(bulk_op, URI, sentences, summarize_rate)
    end_time = time.time()
    logging.debug("sentence process end time : %f" %(end_time))
    logging.debug("sentences total execute time : %f" %(end_time - start_time))
    logging.debug("*******************************\n")


# 프로세스를 시작하는 함수
def process_start(db, mongo, redis, tagger, data):
    logging.debug("*******Process start********")
    start_time = time.time()
    

    try:

        # initialize variable
        contents = []

        # get numpy variable from array of news article
        for (URI, title, content, root_domain, wordcount) in data:
            contents.append(content)
        x_test = get_x_test(contents, tagger)

        # get category_id of each specific classification model
        
        classification_models = get_classification_models()
        
        
        bulk_op = mongo.create_result_bulk_op()
        # start category classification (deep learning)
        ml_classifications(db, mongo, data, x_test, classification_models, bulk_op)

        # keyword/sentence extraction
        keyword(mongo, redis, tagger, data, bulk_op)
        sentence(mongo, redis, tagger, data, bulk_op)
        
        # update read check
        URIs = []
        for (URI, title, content, root_domain, wordcount) in data:
            mongo.bulk_update_metadata(bulk_op, URI, wordcount)
            URIs.append(URI)
        st_time = time.time()
        logging.debug("\n*********************************")
        logging.debug("mongo bulk_op start time : %f" %(st_time))
        mongo.execute_bulk_op(bulk_op)
        en_time = time.time()
        logging.debug("mongo bulk_op start time : %f" %(en_time))
        logging.debug("mongo bulk operation total execution time : %f" %(en_time - st_time))
        logging.debug("***********************************\n")

        
        crawling_bulk_op = mongo.create_crawling_bulk_op()
        mongo.bulk_update_read_check(crawling_bulk_op, URIs)
        stt_time = time.time()
        logging.debug("\n*********************************")
        logging.debug("mongo.execute_bulk_op time : %f" %(stt_time))
        mongo.execute_bulk_op(crawling_bulk_op)
        endd_time = time.time()
        logging.debug("mongo.execute_bulk_op end time : %f" %(endd_time))
        logging.debug("mongo execution total time : %f" %(endd_time - stt_time))
        logging.debug("*************************************\n")

        
        logging.debug("********** Process end ***********")
        elapsed_time = time.time() - start_time
        logging.debug("Process total elapsed time : %f" % (elapsed_time))
        logging.debug("***************************\n")
    except Exception as e:
        logging.debug(traceback.format_exc())
        raise e


# 분산 mod연산
def get_mod_from_server_list(batch, state):
    # get server list in config/config.json
    servers = sorted(load_config()['serverlist'], key=lambda x: x['order'])

    # initialize variables
    server_count = 0
    my_server_idx = -1

    # check servers are alive, and get order of current server
    for idx, server in enumerate(servers):
        try:
            if batch['host'] == server['host'] and batch['port'] == server['port']:
                my_server_idx = server_count
            if check_socket(server['host'], server['port']):
                server_count += 1
        except Exception as e:
            # socket is closed (server down or host is wrong)
            logging.debug(e)
            pass

    if my_server_idx == -1:
        logging.debug('current server does not exist in server list of `config/config.json`')
        # set error flag
        state.value = 1
        return 0, -1

    # 총 서버와 켜져있는 서버 확인
    logging.debug('all server count %d' % (len(servers)))
    logging.debug('running server_count %d' %(server_count))
    return server_count, my_server_idx


# state.value(0: alive, 1: error occur)
def main(state, batch):
    # initialize variables

    try:
        # create database object
        redis = Redis()
        db = DB()
        mongo = Mongo()
        tagger = Mecab()

        # start batch process
        while (True):
            db.ping()
            redis.ping()

            if load_config()['redis']['password']:
                redis.sync_stopwords(db.get_stopwords())
                db.commit()

            # get server count, current server order of all server list
            mod, remainder = get_mod_from_server_list(batch, state)
            
            # error!!! mod cannot be zero
            if mod == 0:
                logging.debug('MOD must be larger than 0. Please check your serverlist of `config/config.json`')
                raise
            
            # check another process is killed
            if state.value == 1:
                break

            # get MOD * 1000 data, and filter
            data = list(map(lambda x: x[1:], list(filter(lambda x: int(x[0], 16) %  mod == remainder, mongo.get_recent_context_data(200 * mod)))))
            logging.debug("data fetched - size: %d" % (len(data)))

            # wait for empty clause
            if len(data) == 0:
                time.sleep(1)
            else:
            # start batch process
                process_start(db, mongo, redis, tagger, data)

    except Exception as e:
        logging.debug(e)
        state.value = 1


def signal_term_handler(signal, frame):
    global state
    global p1
    global p2

    # set process is killed
    state.value = 1
    if p1 and p1.is_alive():
        p1.terminate()

    if p2 and p2.is_alive():
        p2.terminate()

    # kill process
    sys.exit(-1)

def daemon_func():

    # declare global variable
    global state
    global p1
    global p2
    global global_stopwords
    global global_singlewords

    with open(cwd + '/config/stopword.txt') as f:
        global_stopwords = set(map(lambda x: x.strip(), f.readlines()))
    with open(cwd + '/config/singleword.txt') as f:
        global_singlewords = set(map(lambda x: x.strip(), f.readlines()))


    # killing signal handler
    signal.signal(signal.SIGTERM, signal_term_handler)
    config = load_config()['socketio']
    # Multiprocessing Value for shared variable (0: alive, 1: killed)
    state = Value('i', 0)
    try:
        # fork socket server
        p1 = Process(target=server.web_start, args=(state,config['batch']['port']))
        # fork batch server
        p2 = Process(target=main, args=(state,config['batch']))
        p1.start()
        p2.start()

        # check p1 and p2 process is alive
        while p1.is_alive() and p2.is_alive():
            pass
            time.sleep(0.1)
        raise
    except Exception as e:
        # terminate process because daemon process restart the killed process
        if p1 and p1.is_alive():
            p1.terminate()
        if p2 and p2.is_alive():
            p2.terminate()
        # kill process
        sys.exit(-1)
 
if __name__ == '__main__':
    daemon_func()

