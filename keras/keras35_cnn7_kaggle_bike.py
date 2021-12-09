from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
import pandas as pd  
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping


#1)데이터
path = "../_data/kaggle/bike/"      #.지금 현재 작업 공간   / ..이전


train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
#print(x.columns)

test_file = test_file.drop(['datetime'], axis=1)

#print(x.shape)    # (10886, 8)

y = train['count']
#print(y.shape)  #(10886,)

#print(np.unique(y))

import matplotlib.pyplot as plt
import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data= x.corr(), square=True, annot=True, cbar=True)
# plt.show()


#로그변환
y = np.log1p(y)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MaxAbsScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler()

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,2)
x_test = scaler.transform(x_test).reshape(len(x_train),2,2,2)
test_file = scaler.transform(test_file).reshape(len(test_file),2,2,2)

#2) 모델링

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding = 'same', strides=1, input_shape=(2,2,2)))
model.add(Conv2D(2,2))
model.add(Conv2D(20,kernel_size=(3,3)))      
model.add(MaxPooling2D())                                                                                                          
model.add(Flatten())       
model.add(Dense(30))
model.add(Dropout(0.1))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(7))

#3) 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

#4 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

