import pymysql

"""
mysql와 연결
"""
def conn_mysql_fuc():
    conn = pymysql.connect('localhost',user='root',passwd='122345')
    return conn.cursor()

if __name__ == '__main__':
    cursor = conn_mysql_fuc()
    cursor.execute("select * from user")
    res = cursor.fetchall()
    print(res)
