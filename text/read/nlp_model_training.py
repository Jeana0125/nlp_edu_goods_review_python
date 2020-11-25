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
        row = table.row(row=k, columns=['info:keyword'])  # key为相应的row_key，value 即为一整行的数据
        keyword = row[b'info:keyword'].decode().strip()  # 打印多行数据 因为hbase中的数据是二进制的，所以我们进行decode就会转成中文！strip（）去空格的，这个很基础的东西不过多了解了哈哈哈哈
        print(keyword)
