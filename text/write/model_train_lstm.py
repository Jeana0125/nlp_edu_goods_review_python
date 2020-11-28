from utils import pre_processing as pp


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

# 트레이닝 데이터 가져오기
"""
X : 입력 값
y : 레이블
vocab_size : 단어 개수
t : token화 된 결과
max_len: 제일 긴 문장 길이
"""
X, y, vocab_size, t, max_len = pp.get_training_dat(5.0)


model = Sequential()
# 레이블을 분리하였으므로 이제 X의 길이는 max_len-1
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=5, verbose=2)

print("save the model")
model.save("model_lstm_good.h5")
print("done!")





