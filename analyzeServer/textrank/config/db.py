#!/usr/bin/python3

import pymysql
from datetime import datetime
from config import load_config
import sys
import logging

    

class DB():

    def __init__(self):
        mysql = load_config()['mysql']
        try:
            self.conn = pymysql.connect(host=mysql['host'], port=mysql['port'],
                    user=mysql['username'], passwd=mysql['password'], db=mysql['database'], charset='utf8', autocommit=False)
            self.cur = self.conn.cursor()
        except Exception as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))
            raise e

    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    def utcnow(self):
        return datetime.utcnow()

    def ping(self):
        query = "SELECT 1"
        self.cur.execute(query)

    def get_stopwords(self):
        query = " \
            SELECT \
                `COT_EXCL_WORD_INFO`.`EXCL_WORD` AS `word`, \
                `COT_DOMAIN_INFO`.`MAIN_DOMAIN` AS `root_domain` \
            FROM `COT_DOMAIN_EXCL_PLC` \
            INNER JOIN `COT_EXCL_WORD_INFO` ON `COT_DOMAIN_EXCL_PLC`.`EXCL_WORD_SEQ`=`COT_EXCL_WORD_INFO`.`EXCL_WORD_SEQ` AND `COT_EXCL_WORD_INFO`.`EXCL_WORD_USE_YN`='Y' \
            INNER JOIN `COT_DOMAIN_INFO` ON `COT_DOMAIN_EXCL_PLC`.`MEDIA_ID`=`COT_DOMAIN_INFO`.`MEDIA_ID` \
            UNION \
            SELECT \
                `COT_EXCL_WORD_INFO`.`EXCL_WORD` AS `word`, \
                `COT_DOMAIN_EXCL_CTGR_PLC`.`MAIN_DOMAIN` AS `root_domain` \
            FROM `COT_DOMAIN_EXCL_CTGR_PLC` \
            INNER JOIN `COT_EXCL_CTGR_INFO` ON `COT_DOMAIN_EXCL_CTGR_PLC`.`EXCL_CTGR_SEQ`=`COT_EXCL_CTGR_INFO`.`EXCL_CTGR_SEQ` AND `COT_EXCL_CTGR_INFO`.`EXCL_CTGR_USE_YN`='Y' \
            INNER JOIN `COT_EXCL_WORD_INFO` ON `COT_DOMAIN_EXCL_CTGR_PLC`.`EXCL_WORD_SEQ`=`COT_EXCL_WORD_INFO`.`EXCL_WORD_SEQ` AND `COT_EXCL_WORD_INFO`.`EXCL_WORD_USE_YN`='Y'"
        self.cur.execute(query)
        stopwords = []
        for (word, root_domain) in self.cur:
            stopwords.append({"word": word, "root_domain": root_domain})
        return stopwords

    # execute query
    def execute(self, query, params=None):
        self.cur.execute(query, params)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

