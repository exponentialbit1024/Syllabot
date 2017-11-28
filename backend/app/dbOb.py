import pymysql
import gc
gc.collect()

class dbOb:

    def __init__(self):
        self.db = pymysql.connect(MYSQL_HOST,MYSQL_USER,MYSQL_PASS,MYSQL_DB)
