import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim = 1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam    # SGD: 경사하강법

learning_rate = 0.0004

optimizer = Adam(learning_rate= learning_rate)  # optimizer에 learning_rate를 써준다.
# loss:  2.3466 lr: 0.0002 결과물:  [[11.2888]]
#optimizer = Adadelta(learning_rate= learning_rate)
#loss:  2.2552 lr: 0.35 결과물:  [[11.096243]]
#optimizer = Adagrad(learning_rate= learning_rate) 
#loss:  2.6583 lr: 0.041 결과물:  [[11.667839]]
# optimizer = Adamax(learning_rate= learning_rate) 
# loss:  2.8401 lr: 0.03 결과물:  [[11.495453]]
#optimizer = RMSprop(learning_rate= learning_rate) 
#loss:  2.9491 lr: 0.00043 결과물:  [[11.992834]]
#optimizer = SGD(learning_rate= learning_rate) 
#loss:  2.2674 lr: 0.00033 결과물:  [[11.133277]]
#optimizer = Nadam(learning_rate= learning_rate) 
#loss:  2.2725 lr: 0.0004 결과물:  [[11.056805]]

#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse']) 
model.compile(loss = 'mse', optimizer = optimizer)
model.fit(x, y, epochs = 100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size = 1)
y_predict = model.predict([11])

print("loss: ", round(loss,4), 'lr:', learning_rate, '결과물: ', y_predict)