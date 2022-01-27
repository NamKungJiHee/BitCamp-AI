import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import time
### 라벨 축소시킨 것을 다시 smote시켜랏!! ###
#1. 데이터
path = '../_data/winequality/'
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') 
datasets = datasets.values 

x = datasets[:, :11]  
y = datasets[:, 11] 
#print(pd.Series(y).value_counts())   
# 6.0    2198       ## 갯수 ##
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
#print(y)

for index, value in enumerate(y):     #  [20,  163, 1457, 2198,  880,  175,    5]
    if value == 9:
        y[index] = 8
    elif value == 8:
        y[index] = 8
    elif value == 7:
        y[index] = 8
    elif value == 6:
        y[index] = 6
    elif value == 5:
        y[index] = 5
    elif value == 4:
        y[index] = 5
    elif value == 3:
        y[index] = 5
    else:
        y[index] = 0
print(pd.Series(y).value_counts()) 
''' 
6.0    2198 
5.0    1640 
8.0    1060
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8, stratify= y)
#print(pd.Series(y_train).value_counts())   

smote = SMOTE(random_state=66, k_neighbors=1)  # 데이터 증폭 
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
model.score:  0.7041
accuracy_score:  0.7041
f1_score : 0.7007
걸린 시간:  9.7393
'''