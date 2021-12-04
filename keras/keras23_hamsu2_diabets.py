from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=66)   #shuffle은 디폴트가 True

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#print(x.shape)  # (442, 10)
#print(y.shape)    # (442,)

#2) 모델구성
'''
input1 = Input(shape=(10,))  
dense1 = Dense(100,activation = 'relu')(input1)  
dense2 = Dense(80,activation = 'relu')(dense1) 
dense3 = Dense(60)(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)
'''

model = load_model("./_save/keras_02_diabets_save_model.h5")

#model.summary()

#3) 컴파일, 훈련
'''
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode = 'min', verbose=1,restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.11111111, callbacks=[es])
model.save("./_save/keras_02_diabets_save_model.h5")
'''


#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)



# loss:  3884.536865234375
# r2스코어 :  0.511911880856528




"""

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



#그냥
loss:  4867.21630859375
r2스코어 :  0.21879180952387878

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 10)]              0
_________________________________________________________________
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