import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_boston(return_X_y= True)  # return_X_y = True를 쓰면 바로 x와 y를 분리시켜준다.
#print(x.shape, y.shape)  # (506, 13) (506,)

x = np.delete(x,[1,3,6,11],axis=1)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score: ', score)

#print(model.feature_importances_)
''' 
[0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
 0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
 0.4284835 ]
'''
#print(np.sort(model.feature_importances_))  # 오름차순으로 정렬해주기
''' 
[0.00134153 0.00363372 0.01203114 0.01220458 0.01447933 0.01479118
 0.0175432  0.03041654 0.04246344 0.05182539 0.06949984 0.30128664
 0.4284835 ]
'''
aaa = np.sort(model.feature_importances_)  # 오름차순으로 정렬해주기

print("==============================================================================")
for thresh in aaa:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)   # threshold에 낮은 feature_importances 값부터 들어간다. 
                                                                            # ex) 0.03284872이 들어온다면, 0.03 이상인 애들을 모두 선택하라! --> (N,9)  [이렇게 점점 숫자가 줄어듦]
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh = %.3f, n=%d, R2: %2f%%"
          %(thresh, select_x_train.shape[1], score*100))

'''           
(404, 13) (102, 13)
Thresh = 0.001, n=13, R2: 92.211886%
(404, 12) (102, 12)
Thresh = 0.004, n=12, R2: 92.160725%
(404, 11) (102, 11)
Thresh = 0.012, n=11, R2: 92.032829%
(404, 10) (102, 10)
Thresh = 0.012, n=10, R2: 92.197501%
(404, 9) (102, 9)
Thresh = 0.014, n=9, R2: 93.064185%      ### 이때가 가장 성능 good!! ###
(404, 8) (102, 8)
Thresh = 0.015, n=8, R2: 92.363790%
(404, 7) (102, 7)
Thresh = 0.018, n=7, R2: 91.505473%
(404, 6) (102, 6)
Thresh = 0.030, n=6, R2: 92.697396%
(404, 5) (102, 5)
Thresh = 0.042, n=5, R2: 91.777940%
(404, 4) (102, 4)
Thresh = 0.052, n=4, R2: 92.083041%
(404, 3) (102, 3)
Thresh = 0.069, n=3, R2: 92.537126%
(404, 2) (102, 2)
Thresh = 0.301, n=2, R2: 69.849332%
(404, 1) (102, 1)
Thresh = 0.428, n=1, R2: 45.813853%
'''
##############################################################################################
""" 
기존 model.score) 0.9221188601856797
컬럼 제거 후 model.score) 0.9306418465421831
"""