# encoding:utf-8
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from utils import pre_processing as pp

# 모델, 토크나이저, 현재 단어, 반복할 횟수
def sentence_generation(current_word, n,overall):

    model = load_model("model_lstm_good.h5")
    if overall == 3.0:
        model = load_model("model_lstm_normal.h5")
    elif overall == 1.0 or overall == 2.0:
        model = load_model("model_rnn_bad.h5")

    # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    init_word = current_word
    sentence = ''
    sequences_padding, vocab_size, t, max_len = pp.get_padding_data(overall)
    # n번 반복
    for _ in range(n):
        # 현재 단어에 대한 정수 인코딩
        encoded = t.texts_to_sequences([current_word])[0]
        # 데이터에 대한 패딩
        encoded = pad_sequences([encoded], maxlen=max_len, padding='pre')
        # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        result = model.predict_classes(encoded, verbose=0)
        for word, index in t.word_index.items():
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' '  + word
        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence

    return sentence


#print(sentence_generation('good', 20,5.0))
