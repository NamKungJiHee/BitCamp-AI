from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input

#1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=66)   #shuffle은 디폴트가 True

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)  #변환한 값을 다시 x_train에 넣어주자!
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움

#print(x.shape)  #(506, 13)
#print(y.shape)    #(506,)

#2) 모델구성

'''
input1 = Input(shape=(13,))  
dense1 = Dense(10)(input1)  
dense2 = Dense(10, activation='relu')(dense1) 
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
'''

model = load_model("./_save/keras_01_boston_save_model.h5")

#model.summary()


#3) 컴파일, 훈련
'''
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
model.save("./_save/keras_01_boston_save_model.h5")  
'''

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


# loss:  9.380444526672363
# r2스코어 :  0.8864587907608312



"""
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 13)]              0
_________________________________________________________________
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
loss:  28.87620735168457
r2스코어 :  0.6504814589623661
Model: "Sequential"
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