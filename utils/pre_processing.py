import pandas as pd
import numpy as np
import utils.data_utils as du

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


#열 길이 설정
pd.set_option("display.max_colwidth",200)
# 파일 경로
#path = 'C:\\Users\\user\\Desktop\\edu\\reviews_.json'
path = 'D:\\reviews_.json'

"""
파일 읽어오기
"""
def read_file():
    return pd.read_json(path,lines=True)

"""
리뷰 데이터 가져오기 (점수에 근거하여 가져 오기)
"""
def review_data(overall):
    df = read_file()
    # 열 내역 보기
    # print(df.columns)
    # reviewText에 null값이 있는지 판단
    #print(df['reviewText'].isnull().values.any())
    """
    점수에 근거하여 데이터 추출
    """
    # 리스트 선언
    reviewText = []
    if overall == 1.0 or overall == 2.0:
        df_filter = df.loc[df['overall'] == 1.0,['reviewText']]
        df_filter.append(df.loc[df['overall'] == 2.0, ['reviewText']])
        #df_filter = df.loc[df['statisfy'] == 1, ['reviewText']]
        #reveiwText의 값을 저장
        reviewText.extend(list(df_filter.reviewText.values))
    elif overall == 3.0:
        df_filter = df.loc[df['overall'] == 3.0, ['reviewText']]
        # df_filter = df.loc[df['statisfy'] == 2, ['reviewText']]
        # reveiwText의 값을 저장
        reviewText.extend(list(df_filter.reviewText.values))
    else:
        df_filter = df.loc[df['overall'] == 4.0, ['reviewText']]
        df_filter.append(df.loc[df['overall'] == 5.0, ['reviewText']])
        # df_filter = df.loc[df['statisfy'] == 3, ['reviewText']]
        # reveiwText의 값을 저장
        reviewText.extend(list(df_filter.reviewText.values))
    #상위 5개만 출력
    #print(reviewText[:5])
    return reviewText


"""
데이터 추출(소문자화 & 구두점 제거)
"""
def change_review_data(overall):
    text_list =[du.pre_processing(x) for x in review_data(overall)]
    print(len(text_list))
    return text_list


"""
토큰화
"""
def get_token(text):
    t = Tokenizer()
    t.fit_on_texts([text])
    vocab_size = len(t.word_index) + 1
    print("vocab_size: %d" % vocab_size)
    return t, vocab_size


"""
정수 인코딩
"""
def get_encoding_data(overall):
    sequences = list()
    text_list = change_review_data(overall)
    t, vocab_size = get_token(text_list)
    for line in text_list:
        # 각 샘플에 대한 정수 인코딩
        encoded = t.texts_to_sequences([line])[0]
        sequence = list()
        for i in range(1, len(encoded)):
            sequence = encoded[:i + 1]
            sequences.append(sequence)
    return t,vocab_size,sequences


"""
패딩
"""
def get_padding_data(overall):
    index_to_word = {}
    t, vocab_size,sequences = get_encoding_data(overall)

    # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    for key, value in t.word_index.items():
        index_to_word[value] = key
    # 가장 긴 샘플의 길이
    max_len = max(len(l) for l in sequences)
    #print('가장 긴 샘플의 길이: %d' % max_len)
    # padding='pre' : 샘플 길이가 제일 긴 길이보다 짧으면 앞에 0을 추가
    sequences_padding = pad_sequences(sequences, maxlen=max_len, padding='pre')
    #print(sequences[:3])
    return sequences_padding,vocab_size,t,max_len


"""
훈련 데이터 만들기
"""
def get_training_dat(overall):
    sequences_padding,vocab_size,t,max_len = get_padding_data(overall)
    sequences = np.array(sequences_padding)
    X = sequences[:,:-1]
    y_pre = sequences[:,-1]
    #y에 대해 원-핫-인코딩 진행
    y = to_categorical(y_pre,num_classes=vocab_size)
    return X,y,vocab_size,t,max_len


if __name__ == '__main__':
    # 데어터 가져오기
    X,y,vocab_size,t,max_len = get_training_dat(5.0)
    print(X[:3])
    print(y[:3])
