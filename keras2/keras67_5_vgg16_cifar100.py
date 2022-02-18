######## [실습] ########
######## [실습] ########
# 1. vgg trainable : True, False 일 때
# 2. Flatten, GlobalAveragePooling 차이
# 위 4가지 결과와 지금까지 본인이 cifar 10 을 돌려서 최고치가 나온것과 비교!
# loss, accuracy, time 명시!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
import time
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)  
#print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32,32,3)         
x_test = x_test.reshape(10000,32,32,3)           
#print(x_train.shape)     # (50000, 32, 32, 3)
     
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
                                   
x_train=scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)

#2. 모델
vgg16 = VGG16(weights = 'imagenet', include_top= False, input_shape= (32, 32, 3))

vgg16.trainable = False  # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg16)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(100, activation = 'softmax'))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

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
1. vgg_trainable = True / Flatten 일 때

learning_rate:  0.0001
loss:  2.0472
accuracy:  0.5031
걸린시간:  264.7304

2. vgg_trainable = True / GAP 일 때

learning_rate:  0.0001
loss:  2.0425
accuracy:  0.5087
걸린시간:  298.0889

3. vgg_trainable = False / Flatten 일 때

learning_rate:  0.0001
loss:  2.6148
accuracy:  0.3354
걸린시간:  509.6865

4. vgg_trainable = False / GAP 일 때

learning_rate:  0.0001
loss:  2.6123
accuracy:  0.335
걸린시간:  563.8667

## 기존 CNN
loss:  [2.6724672317504883, 0.34599998593330383]
-DNN-
loss :  3.3729944229125977
accuracy :  0.20419999957084656
'''