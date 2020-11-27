import numpy as np
# 시점의 수 즉 문장의 길이.
timesteps = 10
# 입력의 차원. 즉 단어 벡터의 차원.
input_dim = 4
# 은닉 상태의 크기.즉 메모리 셀의 용량.
hidden_size = 8
# 입력에 해당되는 2D 텐서
inputs = np.random.random((timesteps, input_dim))
print(inputs)
# 초기 은닉 상태는 0(벡터)로 초기화
hidden_state_t = np.zeros((hidden_size,))
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
print(hidden_state_t)
# (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wx = np.random.random((hidden_size, input_dim))
# (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size))
# (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).
b = np.random.random((hidden_size,))
print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))
total_hidden_states = []
# 메모리 셀 동작
for input_t in inputs:
  # Wx * Xt + Wh * Ht-1 + b(bias)
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b)
  # 각 시점의 은닉 상태의 값을 계속해서 축적
  total_hidden_states.append(list(output_t))
  # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
  print(np.shape(total_hidden_states))
  hidden_state_t = output_t

# 출력 시 값을 깔끔하게 해준다.
total_hidden_states = np.stack(total_hidden_states, axis = 0)
# (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
print(total_hidden_states)

