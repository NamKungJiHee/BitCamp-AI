#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지 않기!
#3. 래이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. train 70%


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt   #그림
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))


x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(1))

#loss :  432.5167541503906
#r2스코어 :  0.5053462582330829


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

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