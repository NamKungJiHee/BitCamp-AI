import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import time
##### 실습 #####

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

newlist = []
for i in y: 
    #print(i)
    if i <=5:                   
        newlist += [0] 
    elif i ==6:
        newlist += [1]
    else:                      
        newlist += [2]       
y = np.array(newlist)

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
조건1) # 3,4 -> 0 # 5,6,7-> 1 # 8,9 -> 2
#1.그냥 실행 시(SMOTE 전)
model.score:  0.6551
accuracy_score:  0.6551
f1_score : 0.4038      
걸린 시간:  10.7802 

#2. 데이터 증폭(SMOTE 후) / 라벨 축소 전
model.score:  0.6571
accuracy_score:  0.6571
f1_score : 0.4027
걸린 시간:  55.6516

#3. 데이터 증폭(SMOTE 후) / 라벨축소 후
model.score:  0.9276
accuracy_score:  0.9276
f1_score : 0.6141
걸린 시간:  21.1209
===============================================
조건2) # 3,4,5 -> 0 # 6 -> 1 # 7,8,9 -> 2
#1.그냥 실행 시(SMOTE 전)
model.score:  0.6551
accuracy_score:  0.6551
f1_score : 0.4038
걸린 시간:  10.6056

#2. 데이터 증폭(SMOTE 후) / 라벨 축소 전
model.score:  0.6571
accuracy_score:  0.6571
f1_score : 0.4027      
걸린 시간:  56.0612

#3. 데이터 증폭(SMOTE 후) / 라벨축소 후
model.score:  0.7041
accuracy_score:  0.7041
f1_score : 0.7007      
걸린 시간:  9.6648  
'''