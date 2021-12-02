<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
'''
activation(relu) : layer에서 다음 layer로 넘어갈 때 음수는 0으로 해주고 양수만 남겨주는 함수! if 음수값이 사라지면 성능이 조아진다!! 
'''

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation = 'relu')) 
model.add(Dense(4, activation = 'sigmoid'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=30, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''   relu 넣고 돌렸을 때
loss :  1.1436890363693237
4의 예측값 :  [[1.309332]]
=======
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
'''
activation(relu) : layer에서 다음 layer로 넘어갈 때 음수는 0으로 해주고 양수만 남겨주는 함수! if 음수값이 사라지면 성능이 조아진다!! 
'''

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation = 'relu')) 
model.add(Dense(4, activation = 'sigmoid'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=30, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''   relu 넣고 돌렸을 때
loss :  1.1436890363693237
4의 예측값 :  [[1.309332]]
>>>>>>> b6273d91f0d2a8bda64398dfce3bbe5e3e083b07
'''