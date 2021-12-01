from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=66)   #shuffle은 디폴트가 True

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움

#print(x.shape)  # (442, 10)
#print(y.shape)    # (442,)

#2) 모델구성

model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#model.summary()

#3) 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


"""
# 결과
# 그냥
loss:  3095.002197265625
r2스코어 :  0.5032394233838188

# MinMAX
loss:  3282.875732421875
r2스코어 :  0.4730849640753729

#Standard
loss:  2961.40185546875
r2스코어 :  0.5246828887404852

# robust 
loss:  3283.068359375
r2스코어 :  0.47305405004667045

# maxabs
loss:  3119.716064453125
r2스코어 :  0.49927274636480656

== Standard Scaler의 방법으로 돌렸을 때 가장 최소의 loss 값이 나오지만 다른 방법으로 돌렸을 때와 큰(?) 차이가 나지는 않는닷!

##################################### relu 적용 후 
# 그냥
loss:  4996.03125
r2스코어 :  0.1981163947377932

# MinMAX
loss:  3519.0322265625
r2스코어 :  0.4351808962534832

#Standard
loss:  4796.53857421875
r2스코어 :  0.23013582255039366
----------------------------------------위에 3개) relu함수를 중간에 넣어보았더니 넣기 전보다 loss값이 안좋게 나와서 relu의 위치를 끝으로 주어보았다..
# robust 
loss:  3420.1494140625
r2스코어 :  0.45105197063645985

# maxabs
loss:  4280.14599609375
r2스코어 :  0.3130189751200717

==  relu의 값을 끝쪽에 넣어도 loss의 값이 이전보다 안좋게 나왔다!


######################summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 100)               1100
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_2 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_3 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_5 (Dense)              (None, 10)                510
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 11
=================================================================
Total params: 36,971
Trainable params: 36,971
Non-trainable params: 0











"""