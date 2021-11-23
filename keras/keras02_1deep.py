# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))  #(하나의 layer)/ 전:output / 후:input
model.add(Dense(3)) #(어처피 앞에 5가 input이므로 생략)
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

# loss :  0.1437801718711853
# 4의 예측값 :  [[3.1920466]]