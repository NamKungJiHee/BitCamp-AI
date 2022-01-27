import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

#1. 데이터
path = '../_data/winequality/'
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') 
datasets = datasets.values #  pandas --> numpy로 바꿔주기

x = datasets[:, :11]  
y = datasets[:, 11] 
#print(x.shape, y.shape) # (4898, 11) (4898,)
#print(pd.Series(y).value_counts())   
# 6.0    2198       ## 갯수 ##
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8, stratify= y)
# print(pd.Series(y_train).value_counts())   

smote = SMOTE(random_state=66, k_neighbors=1)  # 데이터 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs = 4)

#3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score: ", round(score,4))

y_predict = model.predict(x_test)
print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))
print('f1_score :', round(f1_score(y_test, y_predict, average='macro'),4))

print("====================SMOTE 적용====================")
# smote = SMOTE(random_state=66, k_neighbors=3)  # 데이터 증폭 // k_neighbors(k-최근접이웃 알고리즘)
# x_train, y_train = smote.fit_resample(x_train, y_train) # test는 언제나 원본 그대로여야함! 건들면 안된다!! (평가용)

# #2. 모델
# model = XGBClassifier(n_jobs = 4)

# #3. 훈련
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("model.score: ", round(score,4))

# y_predict = model.predict(x_test)
# print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))
# print('f1_score :', f1_score(y_test, y_predict, average='macro'))
''' 
#그냥 실행 시(SMOTE 전)
model.score:  0.6424
accuracy_score:  0.6424
f1_score : 0.385
===============================================
# 데이터 증폭(SMOTE 후)
model.score:  0.6551
accuracy_score:  0.6551
f1_score : 0.4073
'''