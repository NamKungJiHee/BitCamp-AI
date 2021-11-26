from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)  #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6, shuffle=True, random_state=50)   
x_train, x_val, y_train, y_val = train_test_split(x_test,y_test, train_size=0.5, random_state=50)


model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(120))         
model.add(Dense(110))    
model.add(Dense(100))         
model.add(Dense(1))         

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=200, batch_size=4, validation_data= (x_val, y_val))

loss = model.evaluate(x_test, y_test)   
print('loss : ', loss)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  
print('r2스코어 : ', r2)

#loss :  2389.36767578125
#r2스코어 :  0.6003868813051553