from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np, pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
import time

#1) 데이터

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)   # (581012, 54) (581012,) 
#print(np.unique(y))    # [1 2 3 4 5 6 7]

y = pd.get_dummies(y)   
# print(y.shape)  

x = x.reshape(581012,54,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

print(x_train.shape,x_test.shape)  # (406708, 54, 1) (174304, 54, 1)

scaler = RobustScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape) 


#2) 모델링

model = Sequential()
model.add(Conv1D(80, 2, input_shape = (54,1))) 
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(7, activation ='softmax'))


#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

start= time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 300)
end= time.time()- start
print("걸린시간 : ", round(end,3))

#4 평가예측
loss = model.evaluate(x_test,y_test)

""" 
-LSTM-
loss: 0.2238 - accuracy: 0.9134
-CNN-
loss :  0.03378543630242348
R2 :  0.582790765776973
-DNN-
loss:  [0.5203027725219727, 0.7824496626853943]
r2스코어 :  0.4794340532262525
-Conv1D-
걸린시간 :  160.586
loss: 0.6321 - accuracy: 0.7241
"""



