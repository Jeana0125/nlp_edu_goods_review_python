from utils import pre_processing as pp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 트레이닝 데이터 가져오기
"""
X : 입력 값
y : 레이블
vocab_size : 단어 개수
t : token화 된 결과
max_len: 제일 긴 문장 길이
"""
X, y, vocab_size, t, max_len = pp.get_training_dat(5.0)

"""
모델 설계하기 -- RNN
"""
model = Sequential()
# 레이블을 분리하였으므로 이제 X의 길이는 max_len-1
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# 바닐라 rnn
# hidden layer 수: 32
# return_sequences = True 모든 시점에서 은닉 상태 값을 다음 은닉층에 보내줌
model.add(SimpleRNN(32,return_sequences = True))
# Dense -- fully connected
# Dense(1,input_dim=3,activation='relu') 1은 출력 뉴런의 수, input_dim은 입력 뉴런의 수, activation은 활성화 함수
# activation : linear,sigmoid,softmax,relu
model.add(Dense(vocab_size, activation='relu'))
# optimizer 옵티마이즈 설정 adam,sgd 그라데이션 하강법
# loss 손실 함수
# metrics 모니터링 지표 선택
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 훈련
# epochs 전체 데이터를 훑는 수;
# verbose 출력문구 설정 (0 아무것도 출력 안함 1 훈련 진행도를 보여주는 막대를 보여줌 2 미니 배치마다 손실 정보 출력)
# batch_size 배치크기
# validation_data(x_val, y_val) 검증 데이터, epoch마다 검증데이터의 정확도도 출력, 검증데이터의 loss가 낮아지다가 높아지면 overfitting 현상
model.fit(X, y, epochs=50, verbose=2)


"""
문장 생성 함수
"""
# 모델, 토크나이저, 현재 단어, 반복할 횟수
def sentence_generation(model, t, current_word, n):
    # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    init_word = current_word
    sentence = ''
    # n번 반복
    for _ in range(n):
        # 현재 단어에 대한 정수 인코딩
        encoded = t.texts_to_sequences([current_word])[0]
        # 데이터에 대한 패딩
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
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


print(sentence_generation(model, t, 'good', 20))