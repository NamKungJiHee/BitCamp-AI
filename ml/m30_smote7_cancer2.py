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
# 0    212  --> 112개 삭제
#print(x.shape, y.shape)  # (569, 30) (569,)
''' 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 ....]
'''
index_list = np.where(y==0) # y에서 0이 들어있는 인덱스 위치가 담긴 리스트
print(len(index_list[0])) # 212

del_index_list = index_list[0][100:]
print(len(del_index_list))    # 112

new_x = np.delete(x,del_index_list,axis=0) # del_index_list
new_y = np.delete(y,del_index_list)

x_train,x_test,y_train,y_test = train_test_split(new_x,new_y,shuffle=True, random_state=66, train_size=0.8,stratify=new_y)

smote = SMOTE(random_state = 66, k_neighbors=1)  # 데이터 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)
# print(pd.Series(y).value_counts())
''' 
0    357
1    357
'''
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
# 라벨 0을 112개 삭제해서 재구성하고 smote 전후 비교!!
#1.그냥 실행 시(SMOTE 전)
model.score:  0.9891
accuracy_score:  0.9891
f1_score : 0.9837
걸린 시간:  0.4718

#2. 데이터 증폭(SMOTE 후)
model.score:  0.9674
accuracy_score:  0.9674
f1_score : 0.9529
걸린 시간:  0.6346
'''