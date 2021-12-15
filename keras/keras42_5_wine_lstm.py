from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

#1) 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y) 


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

print(x_train.shape,x_test.shape)   # (124, 13) (54, 13)


scaler = RobustScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(124,13,1)
x_test = scaler.transform(x_test).reshape(54,13,1)


#2) 모델링

model = Sequential()
model.add(LSTM(80, input_shape = (13,1))) 
model.add(Dense(45))
model.add(Dense(3, activation ='softmax'))


#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)


#4 평가예측
loss = model.evaluate(x_test,y_test)

""" 
-LSTM-
loss: 0.4518 - accuracy: 0.8148
-CNN-
loss :  0.004004255402833223
R2 :  0.9833338650668743
-DNN-
loss:  0.2890169620513916
accuracy:  0.8333333134651184
"""