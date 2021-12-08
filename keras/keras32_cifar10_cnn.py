from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten,MaxPooling2D 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) =cifar10.load_data()
print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape         
# x_test = x_test.reshape             
# print(x_train.shape)     

print(np.unique(y_train, return_counts=True))   #  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


x= x_train

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
#print(y_train.shape)  
y_test = to_categorical(y_test)
#print(y_test.shape)

scaler= StandardScaler()
x_train= x_train.reshape(50000,-1)      # 4차원 (50000,32,32,3)을 가로로 1자로 쫙펴준다.  행 세로 열 가로   (50000,3072)
x_test = x_test.reshape(10000,-1)

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)

model = Sequential()
model.add(Conv2D(50,kernel_size=(2,2),strides=1, padding='same', input_shape=(32,32,3))) 
model.add(Conv2D(30,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(30,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',verbose=1, restore_best_weights=True)


model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, callbacks=[es]) 



#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

# loss:  [0.9422957897186279, 0.6801999807357788]

