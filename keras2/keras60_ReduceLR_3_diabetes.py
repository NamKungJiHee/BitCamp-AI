from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

model = Sequential()
model.add(Dense(90, input_dim=10))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(40))         
model.add(Dense(10))         
model.add(Dense(1))         

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss = 'mse', optimizer = optimizer)

es = EarlyStopping(monitor='val_loss', patience=15, mode='min',verbose=1, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
loss = model.evaluate(x_test,y_test)

print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('r2: ', round(r2,4))
print("걸린시간: ", round(end - start,4))

'''
#R2
#0.62 이상

#loss :  2073.370361328125
#r2스코어 :  0.6108767513773117
============================================
# reduce_lr건 결과

learning_rate:  0.001
loss:  2057.9407
r2:  0.6138
걸린시간:  4.2945
'''