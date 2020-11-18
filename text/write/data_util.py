import connect_db.connect_hdfs as cdch

"""
리뷰 쓰기
prepare data
"""

puncs = [']', '[', '（', '）', '{', '}', '：', '!', '...']

def prepare_file():

    files_content = ''

    # hdfs에서 파일 읽어오기
    # pdd = tt.Process_Data_HDFS()
    # content = pdd.connect_hdfs_read("/data/input/test.csv")
    content = cdch.open_file_list("C:\\Users\\user\\Desktop\\edu\\train_sub.csv")
    # print(content)
    for con in content:
        # string.strip() 문구 앞뒤의 공백 제거
        for char in puncs:
            con[2] = con[2].replace(char,"")
            con[1] = con[1].replace(char,"")
        sentence = con[1].strip() + " " + con[2].strip()
        files_content += sentence

    #단어 정열
    words = sorted(files_content.split(" "))
    #print(words)

    #단어 나타난 횟수 카운팅
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 나타난 횟수가 적은 단어 지우기
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    print(counted_words)

    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words,_ = zip(*wordPairs)

    #{'the':1, 'and':2}
    word2num = dict((c, i + 1) for i, c in enumerate(words))
    #{1:'word, 2:'and'}
    num2word = dict((i, c) for i, c in enumerate(words))
    # def word2numF(x):
    #   return word2num.get(x, 0)
    word2numF = lambda x: word2num.get(x, 0)

    #print(word2num)
    #print(num2word)
    #print(words)
    #print(word2numF('and'))

    return word2numF, num2word, words, files_content

if __name__ == '__main__':
    prepare_file()