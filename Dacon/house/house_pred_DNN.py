from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

#1. 데이터
path = '../_data/dacon/housing/'
train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

x = train.drop(['id','target'], axis=1)  
test = test.drop(['id'], axis=1)   
y = train['target']
# print(x.shape, y.shape) # (1350, 15)

le = LabelEncoder()
x['Exter Qual'] = le.fit_transform(x['Exter Qual'])
x['Kitchen Qual'] = le.fit_transform(x['Kitchen Qual'])
x['Bsmt Qual'] = le.fit_transform(x['Bsmt Qual'])
test['Exter Qual'] = le.fit_transform(test['Exter Qual'])
test['Kitchen Qual'] = le.fit_transform(test['Kitchen Qual'])
test['Bsmt Qual'] = le.fit_transform(test['Bsmt Qual'])
# print(x.info())

x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle = True, random_state = 66, train_size = 0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Dense(512,activation = 'relu' ,input_dim=(13)))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16))
model.add(Dense(1, activation='relu'))

#3. 훈련
model.compile(loss = 'mae', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience= 100, restore_best_weights = False)
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 2,
                 validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
y_pred = model.predict(x_test)

#print(y_pred.shape)
y_pred = y_pred.reshape(270,)
#print(y_pred)
print('NMAE_SCORE :' ,NMAE(y_test, y_pred))

# 제출
result = model.predict(test)
#print(result)
submit_file['target']= result
submit_file.to_csv(path + 'sample_submission.csv',index=False)