# 실습
# 아까 4가지로 모델을 만들어랏
# 784개 DNN으로 만든거(최상의 성능인 것//0.9710000157356262)와 비교!

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

scaler =MinMaxScaler()   
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)

pca = PCA(n_components=713)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

pca_EVR = pca.explained_variance_ratio_ 
#print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
#print(cumsum) # 누적합   
#print(np.argmax(cumsum>=0.95) +1) # 154
#print(np.argmax(cumsum)+1) # 713 
#print(np.argmax(cumsum>=0.99)+1) # 331
#print(np.argmax(cumsum>0.999) +1 ) # 486

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape) # (48000, 154) (48000,)
# print(x_test.shape, y_test.shape) # (12000, 154) (12000,)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(80, activation = 'relu', input_shape=(713,)))  
model.add(Dense(60))
model.add(Dropout(0.1))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.25, callbacks=[es]) 
end = time.time()

#4)평가, 예측
#results = model.score(x_test, y_test)
print("걸린시간: ", end - start)
loss = model.evaluate(x_test,y_test)
print("accuracy : ",loss[1])

""" 
1. 나의 최고의 DNN
time = 7.90613055229187
acc = 0.972000002861023

2. 나의 최고의 CNN
time =  76.3782548904419
acc = 0.9825000166893005

3. PCA 0.95
time =  6.115581512451172
acc = 0.9596666693687439

4. PCA 0.99
time =  5.79952597618103
acc = 0.9565833210945129

5. PCA 0.999
time = 4.989681720733643
acc = 0.9482499957084656

6. PCA 1.0
time =   5.871219635009766
acc =  0.9524999856948853
"""