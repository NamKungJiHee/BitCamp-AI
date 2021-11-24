import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
              1.6,1.5,1.4,1.3],
             [10,9,8,7,6,5,4,3,2,1]])

y = np.array([11,12,13,14,15,16,17,18,19,20])


x= x.transpose()
print(x.shape)
print(y.shape)
print(x)


# 모델구성
model = Sequential()
model.add(Dense(3, input_dim=3))  
model.add(Dense(17)) 
model.add(Dense(20))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=30, batch_size=1)


#04. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)
y_predict = model.predict([[10, 1.3, 1]]) 
print('[10, 1.3, 1]의 예측값: ', y_predict)

# loss:  0.01916971430182457
# [10, 1.3, 1]의 예측값:  [[20.282772]]

# mlp뜻
