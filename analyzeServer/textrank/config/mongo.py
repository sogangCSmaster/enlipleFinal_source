#!/usr/bin/python3

import pymongo
from datetime import datetime
from urllib.parse import urlparse
from config import load_config

class Mongo():

    def __init__(self):
        mongo = load_config()['mongo']
        try:
            client = pymongo.MongoClient(mongo['host'], mongo['port'], w=0)
            self.db = client[mongo['database']]
        except (pymongo.errors.ConnectionFailure, e):
            raise e

    def get_object(self):
        return self.db

    # 분석 해야할 기사를 가져온다
    # limit: int
    def get_recent_context_data(self, limit=200):
        data = self.db.crawling.aggregate([
            { "$match": {
                "readCheck": 0,
            }},
            { "$limit": limit },
            { "$sort": { "regDate": 1, "_id": -1 } },
            { "$project": {
                "_id": 1,
                "URI": "$uri",
                "content": "$contents",
                "title": 1,
                "wordCount": 1
            }}
        ])
        return list(map(lambda x: (str(x['_id']), x['URI'], x['title'], x['content'], urlparse(x['URI']).netloc, x['wordCount']), data))

    def bulk_insert_ml_classifications(self, bulk_op, classifications):
        for URI in classifications:
            utcnow = datetime.utcnow()
            bulk_op.find({ 'uri': URI }).upsert().update(
                {
                    "$set": { "classifications": classifications[URI], "modDate": utcnow },
                    "$setOnInsert": { "regDate": utcnow },
                }
            )


    def insert_ml_classifications(self, classifications):

        for URI in classifications:
            utcnow = datetime.utcnow()
            self.db.result.update(
                { "uri": URI },
                {
                    "$set": { "classifications": classifications[URI], "modDate": utcnow },
                    "$setOnInsert": { "regDate": utcnow },
                },
                upsert=True

            )

    def bulk_insert_sentiment(self, bulk_op, URI, result):
        if result == '긍정':
            result = 1
        elif result == '부정':
            result = 5

        utcnow = datetime.utcnow()
        bulk_op.find({ 'uri': URI }).upsert().update(
            {
              "$set": { "likeCode": result, "modDate": utcnow }
            }
        )

    def insert_sentiment(self, URI, result):
        if result == '긍정':
            result = 1
        elif result == '부정':
            result = 5

        utcnow = datetime.utcnow()
        self.db.result.update(
            { "uri": URI },
            {
              "$set": { "likeCode": result, "modDate": utcnow }
            },
            upsert=True
          )

    def create_crawling_bulk_op(self):
        return self.db.crawling.initialize_unordered_bulk_op()
    def create_result_bulk_op(self):
        return self.db.result.initialize_unordered_bulk_op()

    def execute_bulk_op(self, bulk_op):
        return bulk_op.execute()

    def bulk_insert_keywords(self, bulk_op, URI, keywords):
        total = 0.0
        for elem in keywords:
            total += elem[1]
        data = list(map(lambda x: { "contents": x[1][0], "weight": x[1][1], "rank": x[0] + 1, "percent": int(x[1][1] * 100 / total), "rate": None }, enumerate(keywords)))
        utcnow = datetime.utcnow()

        bulk_op.find({ 'uri': URI }).upsert().update(
            {
                "$set": { "keywords": data, "modDate": utcnow },
                "$setOnInsert": { "regDate": utcnow },
            }
        )

    def insert_keywords(self, URI, keywords):
        total = 0.0
        for elem in keywords:
            total += elem[1]
        data = list(map(lambda x: { "contents": x[1][0], "weight": x[1][1], "rank": x[0] + 1, "percent": int(x[1][1] * 100 / total), "rate": None }, enumerate(keywords)))
        utcnow = datetime.utcnow()
        self.db.result.update(
            { "uri": URI },
            {
                "$set": { "keywords": data, "modDate": utcnow },
                "$setOnInsert": { "regDate": utcnow },
            },
            upsert=True
        )

    def bulk_insert_sentences(self, bulk_op, URI, sentences, tr_rate):
        total = 0.0
        for elem in sentences:
            total += elem[1]
        data = list(map(lambda x: { "contents": x[1][0], "weight": x[1][1], "rank": x[0] + 1, "percent": int(x[1][1] * 100 / total), "rate": tr_rate }, enumerate(sentences)))
        utcnow = datetime.utcnow()

        bulk_op.find({ 'uri': URI }).upsert().update(
            {
                "$set": { "sentences": data, "modDate": utcnow },
                "$setOnInsert": { "regDate": utcnow },
            }
        )

    def insert_sentences(self, URI, sentences, tr_rate):
        total = 0.0
        for elem in sentences:
            total += elem[1]
        data = list(map(lambda x: { "contents": x[1][0], "weight": x[1][1], "rank": x[0] + 1, "percent": int(x[1][1] * 100 / total), "rate": tr_rate }, enumerate(sentences)))
        utcnow = datetime.utcnow()
        self.db.result.update(
            { "uri": URI },
            {
                "$set": { "sentences": data, "modDate": utcnow },
                "$setOnInsert": { "regDate": utcnow },
            },
            upsert=True
        )

    def bulk_update_read_check(self, bulk_op, URIs):
        utcnow = datetime.utcnow()
        bulk_op.find({ 'uri' : {  '$in': URIs } }).update(
            { "$set": { "readCheck": 1, "modDate":  utcnow } }
        )


    def update_read_check(self, URI):
        utcnow = datetime.utcnow()
        self.db.crawling.update(
            { "uri": URI },
            { "$set": { "readCheck": 1, "modDate":  utcnow } }
        )

    def bulk_update_metadata(self, bulk_op, URI, wordcount):
        utcnow = datetime.utcnow()
        bulk_op.find({ 'uri': URI }).upsert().update(
            { "$set": { "modDate":  utcnow, "wordCount": wordcount} }
        )

    def update_metadata(self, URI, wordcount):
        utcnow = datetime.utcnow()
        self.db.result.update(
            { "uri": URI },
            { "$set": { "modDate":  utcnow, "wordCount": wordcount} }
        )
