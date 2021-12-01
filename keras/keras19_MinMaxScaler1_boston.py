#######################################
# 각각의 Scaler의 특성과 정의 정리해놓기!
#######################################
# Standard Scaler : 기본 스케일. 평균과 표준편차 사용
# Robust Scaler : 중앙값과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화.
# MaxAbs Scaler : 최대절대값과 0이 각각 1,0이 되도록 스케일링. 절대값이 0~1사이에 매핑되도록 한다. / 큰 이상치에 민감할 수 있음
# MinMax Scaler : 최대/최소값이 각각 1,0이 되도록 스케일링

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#print(np.min(x), np.max(x))  # 최솟값, 최댓값 출력  0.0 711.0
# x = x/711.  # 부동소수점으로 나눈다! 위아래 같음
# x = x/np.max(x) # 전처리   #boston은 컬럼이 13개 이므로 각각 특성마다 최소, 최댓값이 다를것임!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=66)   #shuffle은 디폴트가 True

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)  #변환한 값을 다시 x_train에 넣어주자!
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움

#print(x.shape)  #(506, 13)
#print(y.shape)    #(506,)

#2) 모델구성

model = Sequential()
model.add(Dense(10, activation='relu', input_dim = 13))
model.add(Dense(10, activation= 'relu' ))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

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
loss:  34.7926139831543
r2스코어 :  0.578869070344776

# MinMAX
loss:  16.688833236694336
r2스코어 :  0.797997818741105

#Standard
loss:  17.360122680664062
r2스코어 :  0.7898725287196593

# robust 
loss:  19.352874755859375
r2스코어 :  0.765752201908934

# maxabs
loss:  19.955244064331055
r2스코어 :  0.7584610944352438


== MinMax Scaler의 값이 가장 좋다! loss값이 가장 낮음! 

##################################### relu 적용 후 
# 그냥
loss:  18.9847469329834
r2스코어 :  0.7702080078165496

# MinMAX
loss:  10.857243537902832
r2스코어 :  0.8685835670212866

#Standard
loss:  14.459742546081543
r2스코어 :  0.8249788303231582

# robust 
loss:  15.841588020324707
r2스코어 :  0.8082528958940025

# maxabs
loss:  13.354619979858398
r2스코어 :  0.8383552567483749

==relu를 적용한 후에 전체적으로 loss 값이 확 떨어졌다! 그리고 r2스코어도 이전보다 더 좋아졌다는 것을 알 수 있따!!


######################summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                140
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 481
Trainable params: 481
Non-trainable params: 0

"""

