import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = "../_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values #  pandas --> numpy로 바꿔주기
#print(type(datasets)) # <class 'numpy.ndarray'>

x = datasets[:,:11]  
y = datasets[:, 11]  

#print(x.shape)

#print("라벨: ", np.unique(y, return_counts = True))
# 라벨:  (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 3이 20개, 4가 163개...

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)   
#print(y)
#print(y_train.shape)   
y_test = to_categorical(y_test)
#print(y_test.shape)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (3918, 11) (980, 11) (3918, 10) (980, 10)

#2. 모델
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 11))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=15, mode='min',verbose=1, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

''' 
ML기준!!
#그냥 실행 시(SMOTE 전)
model.score:  0.6424
accuracy_score:  0.6424
f1_score : 0.385
===============================================
# 데이터 증폭(SMOTE 후)
model.score:  0.6551
accuracy_score:  0.6551
f1_score : 0.4073
=======================================
## reduce_lr한 결과 ##
learning_rate:  0.001
loss:  1.0749
accuracy:  0.55
걸린시간:  26.9293
'''