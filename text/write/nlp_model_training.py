import connect_db.connect_hbase as ch

# hbase 연결
con = ch.open_hbase_conn()

#데이터 가져오기
table = con.table("review_satisfy_rpt")

# 데이터 전환
lines = []

def read_data():

    scanner = table.scan()
    for k,v in scanner:
        row = table.row(row=k, columns=['info:keyword'])
        keyword = row[b'info:keyword'].decode().strip()
        print(keyword)
