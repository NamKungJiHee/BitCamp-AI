from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

# print(x_train.shape, x_test.shape)   # (354, 13) (152, 13)

scaler = RobustScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(354,13,1)
x_test = scaler.transform(x_test).reshape(152,13,1)

#2. 모델링
model = Sequential()
model.add(LSTM(80, input_shape = (13,1))) 
model.add(Dense(45))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss = 'mse', optimizer = optimizer) 

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5) 

start = time.time()
model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es, reduce_lr], batch_size = 50)
end = time.time()

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 

#loss, r2 = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('r2: ', round(r2,4))
print("걸린시간: ", round(end - start,4))

"""
-LSTM-
 loss:  16.40131950378418
r2스코어 :  0.8014779145049183   
  
-CNN-
loss :  7.7401814460754395
R2 :  0.906312596952948 

-DNN-
#loss :  23.494508743286133
#r2스코어 :  0.7216934825017536
=========================================
# reduce_lr 걸기 전
loss:  17.07362174987793
learning_rate:  0.001
loss:  17.0736
r2:  0.7933
걸린시간:  9.5029

# reduce_lr 건 후
loss:  13.193345069885254
learning_rate:  0.001
loss:  13.1933
r2:  0.8403
걸린시간:  9.8763
"""