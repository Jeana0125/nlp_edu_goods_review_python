from pyhive import hive
import pandas as pd

"""
hive와 연결
"""
def conn_hive_fuc():
    conn = hive.Connection(host="",port="",username="",password="",database="",auth="LDAP")
    return conn.cursor()


if __name__ == '__main__':
    cursor = conn_hive_fuc()
    cursor.execute("select * from user")
    result = cursor.fetchall()
    df = pd.DataFrame(list(result))

