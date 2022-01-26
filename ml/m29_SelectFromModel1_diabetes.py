import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel

#1. 데이터
x, y = load_diabetes(return_X_y = True)  # return_X_y = True를 쓰면 바로 x와 y를 분리시켜준다.
#print(x.shape, y.shape)  # (442, 10) (442,)

x = np.delete(x,[0,1,4,7],axis=1)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs = -1,
                     n_estimators = 2700, 
                     learning_rate =0.35, 
                     max_depth = 7,         
                     min_child_weight = 1,  
                     subsample = 1,         
                     colsample_bytree = 1, 
                     reg_alpha = 1,      
                     reg_lambda = 0,)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score: ', score)

#print(model.feature_importances_)
''' 
[0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739
 0.07382318 0.03284872 0.39979857 0.06597802]
'''
#print(np.sort(model.feature_importances_))  # 오름차순으로 정렬해주기
''' 
[0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
 0.06597802 0.07382318 0.19681741 0.39979857]
'''
aaa = np.sort(model.feature_importances_)  # 오름차순으로 정렬해주기

print("==============================================================================")
# for thresh in aaa:
#     selection = SelectFromModel(model, threshold = thresh, prefit = True)   # threshold에 낮은 feature_importances 값부터 들어간다.                                                               
#                                                                             # ex) 0.03284872이 들어온다면, 0.03 이상인 애들을 모두 선택하라! --> (N,9)  [이렇게 점점 숫자가 줄어듦]
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs = -1)
#     selection_model.fit(select_x_train, y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
#     print("Thresh = %.3f, n=%d, R2: %2f%%"
#           %(thresh, select_x_train.shape[1], score*100))

'''             # feature_importances가 낮은 순서대로 하나씩 빼면서(n값이 줄어드는 이유) r2_score값을 계산함! Thresh는 feature_importances 반올림 값!
(353, 10) (89, 10)
Thresh = 0.026, n=10, R2: 23.961871%
(353, 9) (89, 9)
Thresh = 0.033, n=9, R2: 27.033263%
(353, 8) (89, 8)
Thresh = 0.038, n=8, R2: 23.870590%
(353, 7) (89, 7)
Thresh = 0.048, n=7, R2: 26.483759%
(353, 6) (89, 6)
Thresh = 0.055, n=6, R2: 30.090053%           ### 이 때 r2 score가 가장 좋다 (즉 feature_importances가 낮은것 기준 앞에 4개를 빼고 돌렸을 때가 성능이 가장 good!!) ###
(353, 5) (89, 5)
Thresh = 0.063, n=5, R2: 27.406967%
(353, 4) (89, 4)
Thresh = 0.066, n=4, R2: 29.839838%
(353, 3) (89, 3)
Thresh = 0.074, n=3, R2: 23.876857%
(353, 2) (89, 2)
Thresh = 0.197, n=2, R2: 14.297310%
(353, 1) (89, 1)
Thresh = 0.400, n=1, R2: 2.562845%
'''
##################################################################################
""" # 0.51이상
기존 model.score)  0.2396187053539609
컬럼 제거 후 model.score) 0.30090053206049094
"""