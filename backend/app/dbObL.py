import pymysql
import gc
gc.collect()
import os
from config import *

class dbOb:

    def __init__(self):
        host = MYSQL_HOST
        user = MYSQL_ROOT
        passw = MYSQL_PASS
        db = MYSQL_DB
        self.db = pymysql.connect(host,user,passw,db)

    def db_conn_close(self):
        self.db.close()
