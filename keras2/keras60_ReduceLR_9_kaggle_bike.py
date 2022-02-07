from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
import time

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
#print(x.columns)

test_file = test_file.drop(['datetime'], axis=1)

#print(x.shape)    # (10886, 8)

y = train['count']
#print(y.shape)  #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

# print(x_train.shape,x_test.shape)  # (7620, 8) (3266, 8)

scaler = MaxAbsScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler()

x_train = scaler.fit_transform(x_train).reshape(7620,8,1)
x_test = scaler.transform(x_test).reshape(3266,8,1)

#2. 모델링
model = Sequential()
model.add(LSTM(80, input_shape = (8,1))) 
model.add(Dense(45))
model.add(Dense(1))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss = 'mse', optimizer = optimizer) 

es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es, reduce_lr], batch_size = 50)
end = time.time()


#4. 평가예측
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('r2: ', round(r2,4))
print("걸린시간: ", round(end - start,4))

""" 
-LSTM-
loss :  21839.546875
r2스코어 0.3336064448238565
-CNN-
loss :  23287.705078125
r2스코어 0.26322775974667534
-DNN-
loss:  [0.5203027725219727, 0.7824496626853943]
r2스코어 :  0.4794340532262525
==================================================
## reduce_lr한 결과 ##

learning_rate:  0.001
loss:  22339.1836
r2:  0.3184
걸린시간:  70.6621
"""