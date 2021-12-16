from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import time

#1) 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

#print(np.unique(y_train))    # [0 1 2 3 4 5 6 7 8 9] = 다중분류

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000,28,28)

scaler = MaxAbsScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler()

x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

#2) 모델링

model = Sequential()
model.add(Conv1D(80, 2, input_shape = (28,28))) 
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(10, activation ='softmax'))


#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

start= time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 300)
end= time.time()- start
print("걸린시간 : ", round(end,3))

#4 평가예측
loss = model.evaluate(x_test,y_test)

"""
-LSTM-
loss: 0.4269 - accuracy: 0.8502
-CNN-
loss:  [0.3179081678390503, 0.8942999839782715]
-DNN-
loss :  0.4216049313545227
accuracy :  0.8529000282287598
-Conv1D-
걸린시간 :  9.786
loss: 0.4456 - accuracy: 0.8434
"""