import pandas as pd
import numpy as np
import utils.data_utils as du
from tensorflow.keras.preprocessing.text import Tokenizer

#열 길이 설정
pd.set_option("display.max_colwidth",200)
# 파일 경로
path = 'C:\\Users\\user\\Desktop\\edu\\reviews_.json'

"""
파일 읽어오기
"""
def read_file():
    return pd.read_json(path,lines=True)


"""
리뷰 데이터 가져오기
"""
def review_data():
    df = read_file()
    # 열 내역 보기
    # print(df.columns)
    # reviewText에 null값이 있는지 판단
    #print(df['reviewText'].isnull().values.any())
    # 리스트 선언
    reviewText = []
    #reveiwText의 값을 저장
    reviewText.extend(list(df.reviewText.values))
    #상위 5개만 출력
    #print(reviewText[:5])
    return reviewText



"""
데이터 추출(소문자화 & 구두점 제거)
"""
def change_review_data():
    text =[du.pre_processing(x) for x in review_data()]
    #print(text[:5])
    return text


def get_training_data():
    review_text_data = change_review_data()
    # vacabulary 크기 확인
    t = Tokenizer()
    t.fit_on_texts(review_text_data)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)
    sequences = list()
    for line in review_text_data:
        encoded = t.texts_to_sequences([line])[0]
        for i in range(1,len(encoded)):
            sequence = encoded[:i+1]
            sequences.append(sequence)
    print(sequences[:11])



if __name__ == '__main__':
    get_training_data()



