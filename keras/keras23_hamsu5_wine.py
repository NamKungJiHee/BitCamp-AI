from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))  
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (178, 3) 


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움



#2) 모델구성
"""
input1 = Input(shape=(13,))  
dense1 = Dense(10)(input1)  
dense2 = Dense(30)(dense1)   
dense3 = Dense(40,activation = 'relu')(dense2)
dense4 = Dense(50,activation = 'relu')(dense3)
output1 = Dense(3,activation = 'softmax')(dense4)
model = Model(inputs=input1, outputs=output1) 
"""

'''
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 13))     
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(3, activation = 'softmax'))
'''
model = load_model("./_save/keras_05_wine_save_model.h5")

#model.summary()

#3) 컴파일, 훈련
"""
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.2,callbacks=[es])

model.save("./_save/keras_05_wine_save_model.h5")
"""

#4) 평가, 예측


loss=model.evaluate(x_test, y_test)
  
print('loss: ', loss[0]) 
print('accuracy: ', loss[1])

'''
results=model.predict(x_test[:7])
print(y_test[:7])  
print(results)
'''


# loss:  0.01570259779691696
# accuracy:  1.0


"""
##################################### relu 적용 후 
# 그냥
loss:  0.5824667811393738
accuracy:  0.7222222089767456

# MinMAX
loss:  0.20618239045143127
accuracy:  0.9722222089767456

#Standard
loss:  0.05545978248119354
accuracy:  0.9722222089767456

# robust 
loss:  0.2698124647140503
accuracy:  0.9722222089767456

# maxabs
loss:  0.01699216477572918
accuracy:  1.0

== wine의 자료의 경우) relu 함수를 쓴 결과 전체적으로 loss의 값이 조아졌다! (그냥 돌렸을 때 빼고!)

######################summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                140
_________________________________________________________________
dense_1 (Dense)              (None, 30)                330
_________________________________________________________________
dense_2 (Dense)              (None, 40)                1240
_________________________________________________________________
dense_3 (Dense)              (None, 50)                2050
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 153
=================================================================
Total params: 3,913
Trainable params: 3,913
Non-trainable params: 0

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 13)]              0
_________________________________________________________________
dense (Dense)                (None, 10)                140
_________________________________________________________________
dense_1 (Dense)              (None, 30)                330
_________________________________________________________________
dense_2 (Dense)              (None, 40)                1240
_________________________________________________________________
dense_3 (Dense)              (None, 50)                2050
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 153
=================================================================
Total params: 3,913
Trainable params: 3,913
Non-trainable params: 0


MaxAbsScaler

loss:  0.0034334834199398756
accuracy:  1.0





"""