import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
#print(x.shape, y.shape) # (178, 13) (178,)
#print(pd.Series(y).value_counts())   
# 1    71     ## 갯수 ##
# 0    59
# 2    48
#print(y)
''' 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
x_new = x[:-30]
y_new = y[:-30]
#print(pd.Series(y_new).value_counts())   
# 1    71
# 0    59
# 2    18
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, random_state=66, shuffle=True, train_size=0.75, stratify= y_new)
print(pd.Series(y_train).value_counts())   
# 1    53
# 0    44
# 2    14

#2. 모델
model = XGBClassifier(n_jobs = 4)

#3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score: ", round(score,4))

y_predict = model.predict(x_test)
print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))

print("====================SMOTE 적용====================")

smote = SMOTE(random_state=66)  # 데이터 증폭
x_train, y_train = smote.fit_resample(x_train, y_train)  # test는 언제나 원본 그대로여야함! 건들면 안된다!! (평가용)

#2. 모델
model = XGBClassifier(n_jobs = 4)

#3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score: ", round(score,4))

y_predict = model.predict(x_test)
print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))

''' 
#그냥 실행 시
model.score:  0.9778 / accuracy_score:  0.9778
===============================================
# 데이터 축소
model.score:  0.9459 / accuracy_score:  0.9459
===============================================
# 데이터 증폭
model.score:  0.973 / accuracy_score:  0.973
'''
