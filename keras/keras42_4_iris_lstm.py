from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split

#1) 데이터



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

# print(x.shape)

#2) 모델링

model = Sequential()
model.add(LSTM(80, input_shape = (4,1))) 
model.add(Dense(45))
model.add(Dense(1, activation =''))


#3. 컴파일, 훈련

model.compile(loss = '', optimizer = 'adam') 

es = EarlyStopping(monitor='', patience=50, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)


#4 평가예측
loss = model.evaluate(x_test,y_test)