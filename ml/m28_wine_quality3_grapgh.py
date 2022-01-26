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

x = datasets.drop(['quality','chlorides','pH'], axis = 1)
y = datasets['quality']

# 라벨:  (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 3이 20개, 4가 163개...

import matplotlib.pyplot as plt
######## 그래프 그리기 ########
count_data = datasets.groupby('quality')['quality'].count()   # quality컬럼을 그룹지을 것. // 갯수
print(count_data)
''' 
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
plt.bar(count_data.index, count_data)  # 세로, 가로
plt.show()
######################################################