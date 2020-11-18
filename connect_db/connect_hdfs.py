from hdfs.client import Client
import csv

txt = []

"""
파일 읽어오기
스트링은 \n으로 나눠짐  
path: 파일 경로
return: line list
"""
def open_file_list(path):
    with open(path,'r',encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            txt.append(row)
    return txt


"""
파일 읽어오기
단어 리스트
path: 파일 경로
return: words list
"""
def open_file_words(path):
    file = open(path, 'r', encoding='utf8')
    for line in file.readlines():
        words = line.split(" ")
        for word in words:
            txt.append(word)
    return txt



class Process_Data_HDFS():

    def __init__(self):
        self.client = Client("http://loacalhost:50070")


    """
    hdfs에서 파일 읽어오기
    return: line list
    """
    def connect_hdfs_read(self,filepath):
        with self.client.read(filepath, encoding='udf-8', delimiter='\n') as fs:
            for line in fs:
                txt.append(line.strip())
        return txt

    """
    hdfs에서 경로 생성
    """
    def connect_hdfs_create(self,path):
        self.client.makedirs(path)


    """
    hdfs에서 파일 업로드
    """
    def connect_hdfs_write(self,path,file_path):
        self.client.upload(path,file_path,cleanup=True)


    """
    hdfs에서 파일 삭제
    """
    def connect_hdfs_delete(self,path):
        self.client.delete(path)


    """
    hdfs에서 파일 로컬에 다운로드
    """
    def connect_hdfs_load(self,hdfs_path,local_path):
        self.client.download(hdfs_path,local_path,overwrite=False)


if __name__ == '__main__':
    #print(open_file_list("C:\\Users\\user\\Desktop\\edu\\word2vec.txt"))
    txt = []
    #print(open_file_words("C:\\Users\\user\\Desktop\\edu\\word2vec.txt"))
    pdd = Process_Data_HDFS()
    print(pdd.connect_hdfs_read())






