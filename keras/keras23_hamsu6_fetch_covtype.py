from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)   # (581012, 54) (581012,) 
#print(np.unique(y))

import pandas as pd   
y = pd.get_dummies(y)   
# print(y.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)  
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2) 모델링
"""
input1 = Input(shape=(54,))  
dense1 = Dense(10,activation = 'relu')(input1)  
dense2 = Dense(10)(dense1)   
dense3 = Dense(10,activation = 'relu')(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(7,activation = 'softmax')(dense4)
model = Model(inputs=input1, outputs=output1) 
"""
'''
model = Sequential()
model.add(Dense(10, activation= 'relu', input_dim = 54))
model.add(Dense(10))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10))
model.add(Dense(7, activation = 'softmax'))
'''
model = load_model("./_save/keras_06_fetch_covtype_save_model.h5")

#model.summary()


#3) 캄파일, 훈련
"""
model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es= EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs = 10000, batch_size=500, validation_split= 0.2,callbacks=[es])  

model.save("./_save/keras_06_fetch_covtype_save_model.h5")
"""

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

# loss:  [0.5053121447563171, 0.7874409556388855]


"""
##################################### relu 적용 후 
# 그냥
loss:  [0.6213434934616089, 0.7356264591217041]

# MinMAX
loss:  [0.5480517148971558, 0.7643434405326843]

#Standard
loss:  [0.5432329773902893, 0.7684139013290405]

# robust 
loss:  [0.527684211730957, 0.7767355442047119]

# maxabs
loss:  [0.5639169812202454, 0.7571319341659546]

== relu함수를 쓰니 그 이전결과보다 loss값이 조아졌음을 볼 수 있따!

######################summary
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                550
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 77
=================================================================
Total params: 957
Trainable params: 957
Non-trainable params: 0

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 54)]              0
_________________________________________________________________
dense (Dense)                (None, 10)                550
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 77
=================================================================
Total params: 957
Trainable params: 957
Non-trainable params: 0


MaxAbsScaler

loss:  [0.5658884644508362, 0.756219744682312]





"""