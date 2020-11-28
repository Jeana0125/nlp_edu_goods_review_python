import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 공부하는 시간
X=np.array([1,2,3,4,5,6,7,8,9])
# 각 공부하는 시간에 맵핑되는 성적
y=np.array([11,22,33,44,53,66,77,87,95])

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# sgd는 경사 하강법. 학습률(learning rate, lr)은 0.01.
sgd = optimizers.SGD(lr=0.01)

# Loss function은 평균제곱오차 mse를 사용
model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])

# 오차를 최소화하는 작업을 300(epochs)번 시도.
model.fit(X,y, batch_size=1, epochs=300, shuffle=False)

# 그래프로 보기
import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X,y, 'k.')

#9.5시간 공부하면 몇점인지 예측
print(model.predict([9.5]))