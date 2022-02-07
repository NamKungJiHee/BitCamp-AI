import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)  # 흑백이미지   6만장의 이미지가 28,28이다...
#print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)  

x_train = x_train.reshape(60000,28,28,1)/255.           #(60000,28,14,2)도 가능 / reshape= 위에 train,test의 값과 맞춰주는것!
x_test = x_test.reshape(10000,28,28,1)/255.             # ex) (60000, 28, 28) 가로*세로*장수 ==>   (60000,28,28,1)  
#print(x_train.shape)     # (60000, 28, 28, 1)

#print(np.unique(y_train, return_counts=True))   # [0 1 2 3 4 5 6 7 8 9]  return_counts=True) 하면 pandas의 value.counts와 같은 기능

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
print(y_train.shape)   #(60000, 10)
y_test = to_categorical(y_test)
print(y_test.shape)

#2. 모델

model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])  #sparse_categorical_crossentropy

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.25, callbacks=[es]) 
end = time.time()

#4. 평가, 예측

loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

''' 
# 0.01일 때와 0.001, 0.0001일 때 성능과 시간 비교하기!!

learning_rate: 0.01
loss:  0.2015
accuracy:  0.9405
걸린시간:  100.9711

learning_rate:  0.001     # Adam의 default값!! 0.001
loss:  0.0526
accuracy:  0.9843
걸린시간:  99.5855

learning_rate:  0.0001
loss:  0.0596
accuracy:  0.9816
걸린시간:  100.9733
'''