# 증폭해서 저장한 데이터를 불러와서 완성할것!
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_covtype
from xgboost import XGBClassifier
##### 실습 #####
#1. 데이터

datasets = fetch_covtype() 
x = datasets.data
y = datasets.target
#print(x.shape, y.shape) # (581012, 54) (581012,)
# print(pd.Series(y).value_counts())   
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

smote = SMOTE(random_state = 66, k_neighbors=1)  
x_train, y_train = smote.fit_resample(x_train, y_train)

#print(x_train.shape, y_train.shape) # (1587579, 54) (1587579,)
#print(pd.Series(y_train).value_counts()) 
''' 
1    226797
2    226797
6    226797
3    226797
7    226797
5    226797
4    226797
'''
x_train = np.load('D:\\Study\\_save\\_fetch_covtype_x_train.npy')
y_train = np.load('D:\\Study\\_save\\_fetch_covtype_y_train.npy')
#x_test = np.load('D:\\Study\\_save\\_fetch_covtype_x_test.npy')
#y_test = np.load('D:\\Study\\_save\\_fetch_covtype_y_test.npy')

#2. 모델, 훈련
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9,
#               enable_categorical=False, eval_metric='merror', gamma=0,
#               gpu_id=-1, importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=6,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=300, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', predictor= 'gpu_predictor', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None)

#######################################################################
# n_splits = 3
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parameters = [
#     {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
#     {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},]
# model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True)


import joblib
model = joblib.load('D:\\Study\\_save\\_result.dat')

import joblib
model = joblib.load('D:\\Study\\_save\\_result1.dat')

#3. 훈련
start = time.time()
# model.fit(x_train, y_train)
end = time.time()

score = model.score(x_test, y_test)
print("model.score: ", round(score,4))

y_predict = model.predict(x_test)
print("accuracy_score: ", round(accuracy_score(y_test, y_predict),4))
print('f1_score :', round(f1_score(y_test, y_predict, average='macro'),4))
print("걸린 시간: ", round(end-start,4))

# import joblib
# joblib.dump(model,"D:\\Study\\_save\\_result.dat")

# import joblib
# joblib.dump(model,"D:\\Study\\_save\\_result1.dat")

'''  load 성공!! (Standard로 썼을 때)
model.score:  0.9141
accuracy_score:  0.9141
f1_score : 0.9121
걸린 시간:  36.3072

(PowerTransformer로 돌렸을 때)
model.score:  0.9143
accuracy_score:  0.9143
f1_score : 0.9107
걸린 시간:  62.8455

(RandomizedSearchCV로 돌렸을 때)
model.score:  0.964
accuracy_score:  0.964
f1_score : 0.948
'''