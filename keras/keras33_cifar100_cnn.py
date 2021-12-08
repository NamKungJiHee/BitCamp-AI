from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten,MaxPooling2D 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split

#1.데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) =cifar100.load_data()

#print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)  

print(np.unique(y_train, return_counts=True))  
 
x= x_train

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
print(y_train.shape)  
y_test = to_categorical(y_test)
print(y_test.shape)

scaler= StandardScaler() #MinMaxScaler() RobustScaler() MaxAbsScaler()

"""
x_train= x_train.reshape(50000,-1)
x_test = x_test.reshape(10000,-1)

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)
"""


x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

#2.모델링
model = Sequential()
model.add(Conv2D(25,kernel_size=(2,2),strides=1, padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(35,(3,3), activation='relu'))
model.add(Conv2D(45,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(30))
model.add(Dense(100, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',verbose=1, restore_best_weights=True)


model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, callbacks=[es]) 



#4평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

#  loss:  [2.6724672317504883, 0.34599998593330383]
