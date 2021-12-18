from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.convolutional import Conv1D 
import time

#1) 데이터

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

x = x.to_numpy()
x = x.reshape(10886,2,4)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

print(x_train.shape,x_test.shape)  # (7620, 2, 4) (3266, 2, 4)

scaler = MaxAbsScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler()

x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)


# #2) 모델링

model = Sequential()
model.add(Conv1D(80, 2, input_shape = (2,4))) 
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam') 

es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)

start= time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)
end= time.time()- start
print("걸린시간 : ", round(end,3))

#4 평가예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)


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
-Conv1D-
걸린시간 :  7.373
loss :  24268.296875
r2스코어 0.25949764040407874
"""