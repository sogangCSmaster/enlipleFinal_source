#!/usr/bin/python3

import redis
import json
from config import load_config

class Redis():

    def __init__(self):
        redis_config = load_config()['redis']
        try:
            if redis_config['password']:
                self.r = redis.StrictRedis(host=redis_config['host'], port=redis_config['port'],password=redis_config['password'], db=0)
            else:
                self.r = redis.StrictRedis(host=redis_config['host'], port=redis_config['port'], db=0)
        except ConnectionError as e:
            raise e

    def set(self, key, value):
        return self.r.set(key, value)

    def get(self, key):
        return self.r.get(key)  

    def ping(self):
        return self.r.ping()

    def sync_stopwords(self, stopwords=[]):
        self.set('stopwords', json.dumps(stopwords, separators=(',',':')))

    def get_stopwords(self, root_domain):
        data = self.get('stopwords')

        # empty key handling
        if data is None:
            return set([])
        data = data.decode('utf-8')
        j = json.loads(data)

        stopwords = []

        for x in j:
            if x['root_domain'] is None or x['root_domain'] == root_domain:
                stopwords.append(x)
        return set(map(lambda x: x['word'], stopwords))
