import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
##### 실습 #####

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
#print(pd.Series(y).value_counts())   
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8, stratify= y)
#print(pd.Series(y_train).value_counts())   
#print(np.unique(y_train,return_counts=True))
smote = SMOTE(random_state=66, k_neighbors=8)  # 데이터 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs = 4, n_estimators=2000,
    learning_rate = 0.025,
    max_depth = 5,
    min_child_weight = 1,
    subsample =1,
    colsample_bytree =1,
    reg_alpha =1,              
    reg_lambda=0)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

score = model.score(x_test, y_test)
print("model.score: ", round(score,4))

y_predict = model.predict(x_test)
print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))
print('f1_score :', round(f1_score(y_test, y_predict, average='macro'),4))
print("걸린 시간: ", round(end-start,4))

''' 
#1.그냥 실행 시(SMOTE 전)
model.score:  0.9649
accuracy_score:  0.9649
f1_score : 0.9619
걸린 시간:  0.5306

#2. 데이터 증폭(SMOTE 후)
model.score:  0.9737
accuracy_score:  0.9737
f1_score : 0.9713
걸린 시간:  0.6572
'''