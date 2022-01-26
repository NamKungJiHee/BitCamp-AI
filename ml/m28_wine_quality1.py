import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
### 평가지표 acc ###

### 방법 1 ###
"""
#1. 데이터
path = "../_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리
#print(datasets.shape)  # (4898, 12)
#print(datasets.head())
''' 
   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
0            7.0              0.27         0.36            20.7      0.045                 45.0                 170.0   1.0010  3.00       0.45      8.8        6
1            6.3              0.30         0.34             1.6      0.049                 14.0                 132.0   0.9940  3.30       0.49      9.5        6
2            8.1              0.28         0.40             6.9      0.050                 30.0                  97.0   0.9951  3.26       0.44     10.1        6
3            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
4            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
'''
#print(datasets.describe())  # describe는 수치데이터 
''' 
       fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000
mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909
std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639
min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000
25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000
50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000
75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000
max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000
'''
#print(datasets.info())
''' 
 #   Column                Non-Null Count  Dtype
---  ------                --------------  ----- 
 0   fixed acidity         4898 non-null   float64                  # non-null : 결측치 x
 1   volatile acidity      4898 non-null   float64
 2   citric acid           4898 non-null   float64
 3   residual sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free sulfur dioxide   4898 non-null   float64
 6   total sulfur dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64
dtypes: float64(11), int64(1)
'''
x = datasets.drop(['quality','chlorides','pH'], axis = 1)
y = datasets['quality']
# print(x.shape, y.shape)  #(4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_estimators = 2500, learning_rate =0.125, n_jobs = -1, 
                     max_depth = 12,        
                     min_child_weight = 1,  
                     subsample = 1,        
                     colsample_bytree = 1,  
                     reg_alpha = 1,     
                     reg_lambda = 0,)
                                                             
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1,
          eval_set = [(x_test, y_test)],  
          eval_metric= 'mlogloss',
          early_stopping_rounds= 30)
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", round(results,4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", round(acc,4))
"""

''' 
걸린시간:  4.262447118759155
results:  0.7235
acc:  0.7235
'''

#######################################################################################################################
### 방법 2 ###

#1. 데이터
path = "../_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values #  pandas --> numpy로 바꿔주기
#print(type(datasets)) # <class 'numpy.ndarray'>

x = datasets[:,:11]  # 모든 행, 10번째까지
y = datasets[:, 11]  # 모든행, 11번째 열이 y 

print("라벨: ", np.unique(y, return_counts = True))
# 라벨:  (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 3이 20개, 4가 163개...

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score: ', score)
print('acc_score: ', accuracy_score(y_test, y_predict))
print('f1_score1: ', f1_score(y_test, y_predict, average = 'macro'))  
print('f1_score2: ', f1_score(y_test, y_predict, average = 'micro'))  

# f1 score는 비율이 다를 때 (데이터 양이 불균형일 때 사용) #  
# ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
''' 
model.score:  0.6591836734693878
acc_score:  0.6591836734693878
f1_score1:  0.41005452777318885
f1_score2:  0.6591836734693878
'''