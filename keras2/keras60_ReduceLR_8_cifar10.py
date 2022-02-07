from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten,MaxPooling2D 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) =cifar10.load_data()
print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape         
# x_test = x_test.reshape             
# print(x_train.shape)     
print(np.unique(y_train, return_counts=True))   #  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
#print(y_train.shape)  
y_test = to_categorical(y_test)
#print(y_test.shape)

scaler= StandardScaler()              
x_train= x_train.reshape(50000,-1)   
x_test = x_test.reshape(10000,-1)    
                                   
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)

#2. 모델
model = Sequential()
model.add(Conv2D(50,kernel_size=(2,2),strides=1, padding='same', input_shape=(32,32,3))) 
model.add(Conv2D(30,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(30,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5) # 5번 동안에 갱신이 안되면 50%의 learning_rate를 감소시키겠다.(그래서 처음 learning_rate는 범위를 크게 잡는 것이 good)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

'''
기존 CNN
# loss:  [0.9422957897186279, 0.6801999807357788]
========================================================
# reduce_lr를 한 결과)

learning_rate:  0.001
loss:  0.9223
accuracy:  0.6798
걸린시간:  68.5412
'''