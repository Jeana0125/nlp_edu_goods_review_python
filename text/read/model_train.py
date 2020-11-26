import pandas as pd
import numpy as np
import utils.data_utils as du
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM


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
        #sequence = encoded[:i+1]
        sequences.append(encoded[:i+1])
#print(sequences[:11])
index_to_word=[]
for key,value in t.word_index.items():
    index_to_word[value] = key
print('빈도수 상위 582번 단어 : {}'.format(index_to_word[582]))
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])
sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
print(X[:3])
print(y[:3])
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)


def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items():
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, 'how', 10))





