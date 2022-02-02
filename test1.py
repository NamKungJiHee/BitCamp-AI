#1. fetch california data load하여 x,y나누고 x에 칼럼 pandas형식으로 칼럼이름 넣어주고,
#2. xgboost default값으로 모델 돌려서 model.score 출력하고
#3. Feature Importances구해서 낮은 순서대로 어떤 칼럼이 중요도가 몇인지 확인할 수 있게 출력하고
#4. 직접 함수를 만들거나 modelselect이용해서 칼럼수 1개씩 계속 줄여보면서 xgboost default값으로 계속 돌려보면서 정확도 비교.
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

pd.set_option('display.max_columns',10)
pd.set_option('display.width',150)

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

# print(x[:5])
#print(x.shape, y.shape) # (20640, 8) (20640,)

#print(type(x)) # <class 'numpy.ndarray'>
x = np.delete(x,[0],axis=1)
#print(x.shape, y.shape) # (20640, 7) (20640,)
# #x = np.delete(x,[0,1],axis=1)

# x = pd.DataFrame(x)
# x = x.drop(['AveOccup'], axis = 1)
#print(x.shape, y.shape)

#print(x.shape, y.shape) # (20640, 8) (20640,)
#print(datasets.feature_names) # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# x = pd.DataFrame(x)
# feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# x.columns = feature_names
# # print(x[:5])

df = pd.DataFrame(x, columns=[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']])  # x에 컬럼 이름을 넣어주기
#print(df)
''' 
       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
0      8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
1      8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
2      7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
3      5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
4      3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2.  모델
model = XGBRegressor()

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
# parameters = [{'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]}]
# model = GridSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1, refit=True)

#3. 훈련 
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 
results = model.score(x_test, y_test)
print("results: ", round(results, 4))
# results:  0.8434

#print(model.feature_importances_)
# [0.45402217 0.07616177 0.04781106 0.02427038 0.02587491 0.15818097 0.10008781 0.11359099]

#3. Feature Importances구해서 낮은 순서대로 어떤 칼럼이 중요도가 몇인지 확인할 수 있게 출력하고
bbb = pd.DataFrame([model.feature_importances_], columns=[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']])
#print(bbb)
''' 
     MedInc  HouseAge  AveRooms AveBedrms Population  AveOccup  Latitude Longitude
0  0.454022  0.076162  0.047811   0.02427   0.025875  0.158181  0.100088  0.113591
'''

bbb= np.sort(model.feature_importances_)  
#print(bbb)
''' 
[0.02427038 0.02587491 0.04781106 0.07616177 0.10008781 0.11359099
 0.15818097 0.45402217]
'''
#4. 직접 함수를 만들거나 modelselect이용해서 칼럼수 1개씩 계속 줄여보면서 xgboost default값으로 계속 돌려보면서 정확도 비교.

# for thresh in bbb:
#     selection = SelectFromModel(model, threshold = thresh, prefit = True)   
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs = -1)
#     selection_model.fit(select_x_train, y_train, eval_metric='merror')
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
#     print("Thresh = %.3f, n=%d, R2: %2f%%"
#           %(thresh, select_x_train.shape[1], score*100))
  
''' 
(16512, 8) (4128, 8)
Thresh = 0.024, n=8, R2: 84.339004%
(16512, 7) (4128, 7)
Thresh = 0.026, n=7, R2: 84.710524%  --> good (1개 삭제)
(16512, 6) (4128, 6)
Thresh = 0.048, n=6, R2: 84.638097%
(16512, 5) (4128, 5)
Thresh = 0.076, n=5, R2: 83.316469%
(16512, 4) (4128, 4)
Thresh = 0.100, n=4, R2: 82.830756%
(16512, 3) (4128, 3)
Thresh = 0.114, n=3, R2: 72.019317%
(16512, 2) (4128, 2)
Thresh = 0.158, n=2, R2: 59.743375%
(16512, 1) (4128, 1)
Thresh = 0.454, n=1, R2: 50.002730%
'''

########
# 그냥 돌렸을 때 : 0.8434
# 컬럼 삭제 후 : 0.8303
########