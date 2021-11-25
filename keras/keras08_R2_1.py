## R2가 먼지 찾아라!
#R2 값은 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것. (0.9999면 Good)
# 1은 예측이 완벽한 경우고, 0은 훈련 세트의 출력값인 y_train의 평균으로만 예측하는 모델의 경우
#회귀모델: 어떤 자료에 대해서 그 값에 영향을 주는 조건을 고려하여 구한 평균
#공식: R2 = 1 - 오차2 / 편차2


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt   #그림
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])


x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(170))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=150, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   #결과확인용/ 영향 안미침..
print('loss : ', loss)
# loss :  23.159833908081055

y_predict = model.predict([x_test])

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  #loss의 보조지표
print('r2스코어 : ', r2)
#loss :  6.9489827156066895
#r2스코어 :  0.5114000988953997


#loss :  22.81608009338379
#r2스코어 :  -1.178723820434044

#plt.scatter(x,y)     #점 찍어주기
#plt.plot(x, y_predict, color= 'red')  #선 그려주기
#plt.show()
