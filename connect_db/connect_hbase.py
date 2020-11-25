import happybase

"""
hbase 설치한 url과 default port 9090
"""
def open_hbase_conn():
    conn = happybase.Connection("192.168.101.11",9090)
    return conn


if __name__ == '__main__':
    print(open_hbase_conn().tables())