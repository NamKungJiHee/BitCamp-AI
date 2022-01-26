from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  
from sklearn.preprocessing import PowerTransformer      
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score, r2_score, accuracy_score
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#1. 데이터 
path = '../_data/winequality/'
datasets = pd.read_csv(path +"winequality-white.csv", header = 0, sep=';')
#print(datasets.shape) #(4898, 12)

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    ### IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치) ###
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

# 이상치 제거한 데이터셋
datasets = remove_outlier(datasets)
#print(datasets[:40])

# 이상치 채워주기
datasets = datasets.interpolate()
#print(datasets[:40])
############################################################################################
'''
x = datasets.drop(['quality'], axis=1)
y = datasets['quality']
#print(x.shape, y.shape) #(4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8, stratify = y)

print(x_train.shape, y_train.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=10000,
    learning_rate = 0.4,
    max_depth = 6,
    min_child_weight = 0.9,
    subsample =1,
    colsample_bytree =0.9,
    reg_alpha =1,              
    reg_lambda=0,              
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor',)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',          
          early_stopping_rounds=20
          )
end = time.time()

print( "걸린시간 :", end - start)

#4. 평가
results = model.score(x_test, y_test) 
print("results :", results)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", acc)
print('f1_score :', f1_score(y_test, y_pred, average='micro'))
'''