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
X, y, vocab_size, t, max_len = pp.get_training_dat(1.0)

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
model.fit(X, y, epochs=200, verbose=2)

print("save the model")
model.save("model_rnn_bad.h5")
print("done!")