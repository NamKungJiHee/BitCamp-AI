import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#01. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
              1.6,1.5,1.4,1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])


#x= x.reshape(10,2)   행과 열 change 앞에서부터 묶어줌../ 데이터 순서 안바뀜..
x= x.transpose()  #행과열 change 가로와 세로가 바뀜.. 


print(x.shape)
print(y.shape)


#x = x.T
print(x)


# 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))  
model.add(Dense(15)) 
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x ,y , epochs=30, batch_size=1)


#04. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)
y_predict = model.predict([[10, 1.3]]) #열 우선 행 무시..
print('[10, 1.3]의 예측값: ', y_predict)

#loss:  3.677257537841797
#[10, 1.3]의 예측값:  [[21.53298]]