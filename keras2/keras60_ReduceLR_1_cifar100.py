import numpy as np
from tensorflow.keras.datasets import cifar100 
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D, Flatten, GlobalAveragePooling2D
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)  
#print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32,32,3)/255.         
x_test = x_test.reshape(10000,32,32,3)/255.           
#print(x_train.shape)     # (50000, 32, 32, 3)

#print(np.unique(y_train, return_counts=True))   # [0 1 2 3 4 5 6 7 8 9]  return_counts=True) 하면 pandas의 value.counts와 같은 기능

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
print(y_train.shape)  
y_test = to_categorical(y_test)
print(y_test.shape)

#2. 모델
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(32,32,3)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())   
#model.add(GlobalAveragePooling2D()) 
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])  #sparse_categorical_crossentropy

es = EarlyStopping(monitor='val_loss', patience=15, mode='min',verbose=1, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5) # 5번 동안에 갱신이 안되면 50%의 learning_rate를 감소시키겠다.(그래서 처음 learning_rate는 범위를 크게 잡는 것이 good)

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

'''
## 기존 결과 ##
-LSTM-
loss: 3.2097 - accuracy: 0.2342
-CNN-
loss:  [2.6724672317504883, 0.34599998593330383]
-DNN-
loss :  3.3729944229125977
accuracy :  0.20419999957084656
====================================================
## reduce_lr한 결과 ##

# GlobalAveragePooling2D #
learning_rate:  0.001
loss:  3.3062
accuracy:  0.1917
걸린시간:  499.2555

# Flatten #
learning_rate:  0.001
loss:  3.0104
accuracy:  0.2661
걸린시간:  585.7514
'''