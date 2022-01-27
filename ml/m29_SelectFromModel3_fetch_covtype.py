import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
# RandomizedSearchCV 적용해서 출력한 값에서 feature_importances 추출후 selectFromModel 만들어서 컬럼 축소 후 모델구축해서 결과 도출! #

#1. 데이터
datasets = fetch_covtype() 
x = datasets.data
y = datasets.target
#print(dataset.feature_names)
x = np.delete(x,[20, 28],axis=1)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},]
model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score: ', score)
''' 
model.score: 
'''
print(model.best_estimator_.feature_importances_)
''' 
[0.05022869 0.00855187 0.00591726 0.0090965  0.01164381 0.01455205
 0.00957581 0.00978856 0.01211412 0.01567931 0.06275129 0.01683292
 0.09607214 0.00810047 0.00452978 0.00989864 0.01094213 0.01530006
 0.00412852 0.00329604 0.         0.00116131 0.00162447 0.01106238  
 0.02001779 0.00602387 0.00940748 0.00922188 0.00066598 0.00351829
 0.03000731 0.00093116 0.00246228 0.00520531 0.00651642 0.0353939 
 0.02479506 0.01201034 0.00106216 0.00209699 0.00711565 0.0031447
 0.00785613 0.01529509 0.0329822  0.1436936  0.0349775  0.00316917
 0.04652174 0.00296806 0.03082643 0.02683516 0.06091644 0.02151388]
'''
print(np.sort(model.best_estimator_.feature_importances_))  # 오름차순으로 정렬해주기
aaa = np.sort(model.best_estimator_.feature_importances_)  # 오름차순으로 정렬해주기

print("==============================================================================")
for thresh in aaa:
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)   
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train, eval_metric='merror')
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh = %.3f, n=%d, R2: %2f%%"
          %(thresh, select_x_train.shape[1], score*100))

'''         
(464809, 54) (116203, 54)
Thresh = 0.001, n=54, R2: 74.352787%
(464809, 53) (116203, 53)
Thresh = 0.003, n=53, R2: 74.732154%
(464809, 52) (116203, 52)
Thresh = 0.003, n=52, R2: 74.749469%     ### 성능 제일 good ###
(464809, 51) (116203, 51)
Thresh = 0.003, n=51, R2: 74.431714%
(464809, 50) (116203, 50)
Thresh = 0.004, n=50, R2: 73.706991%
(464809, 49) (116203, 49)
Thresh = 0.004, n=49, R2: 73.541589%
(464809, 48) (116203, 48)
Thresh = 0.004, n=48, R2: 73.541589%
...........
'''
##################################################################################
""" 
기존 model.score) 0.9723070832938908
컬럼 제거 후 model.score) 0.9724619846303452
"""
