from random import random
import numpy as np, pandas as pd
import time
from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])  # percentile : 백분위수  
                                                                       
    print("1사분위: ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3 - quartile_1  
    print("iqr: ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

path = '../_data/dacon/housing/' 
datasets = pd.read_csv(path + 'train.csv', index_col = 0, header = 0)
#print(datasets)
test_sets = pd.read_csv(path + 'test.csv', index_col = 0, header = 0)

submit_sets = pd.read_csv(path + 'sample_submission.csv', index_col = 0, header = 0)
#print(datasets.info())
''' 
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   Overall Qual    1350 non-null   int64
 1   Gr Liv Area     1350 non-null   int64
 2   Exter Qual      1350 non-null   object            Object = String과 비슷
 3   Garage Cars     1350 non-null   int64
 4   Garage Area     1350 non-null   int64
 5   Kitchen Qual    1350 non-null   object
 6   Total Bsmt SF   1350 non-null   int64
 7   1st Flr SF      1350 non-null   int64
 8   Bsmt Qual       1350 non-null   object
 9   Full Bath       1350 non-null   int64
 10  Year Built      1350 non-null   int64
 11  Year Remod/Add  1350 non-null   int64
 12  Garage Yr Blt   1350 non-null   int64
 13  target          1350 non-null   int64
'''
#print(datasets.describe())    # 수치형만 보여줌
#print(datasets.isnull().sum())   # Null값 없음 

#################### 중복값 처리 ####################
print("중복값 제거 전: ", datasets.shape) # 중복값 제거 전:  (1350, 14)
datasets = datasets.drop_duplicates()
print("중복값 제거 후: ", datasets.shape) # 중복값 제거 후:  (1349, 14)

#################### 이상치 확인 처리 ####################
outliers_loc = outliers(datasets['Garage Yr Blt'])
print(outliers_loc)
''' 
1사분위:  1961.0
q2:  1978.0
3사분위:  2002.0
iqr:  41.0
(array([254], dtype=int64),)  # 254자리가 문제있다는 뜻!!!
'''
print(datasets.loc[[255], 'Garage Yr Blt'])  #  2207
''' 
id
255    2207
Name: Garage Yr Blt, dtype: int64
'''
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace = True)
#print(datasets.shape)  # (1348, 14)

#print(datasets['Exter Qual'].value_counts())
''' 
TA    808
Gd    485
Ex     49
Fa      8
'''
#print(datasets['Kitchen Qual'].value_counts())
''' 
TA    660
Gd    560
Ex    107
Fa     23
'''
#print(datasets['Bsmt Qual'].value_counts())
''' 
TA    605
Gd    582
Ex    134
Fa     28
Po      1
'''
#print(test_sets['Exter Qual'].value_counts())
#print(test_sets['Kitchen Qual'].value_counts())  # Po 1개
#print(test_sets['Bsmt Qual'].value_counts())      # Po 1개

########################라벨인코더 대신 수동인코더 해준다.########################
qual_cols = datasets.dtypes[datasets.dtypes == np.object].index      # datasets에서 object인 것들을 qual_cols로 #
#print(qual_cols)  # Index(['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'], dtype='object')

# 품질 관련 변수 → 숫자로 매핑 // Poor(Po) → Fa(Fair) →Typical/Average(TA)→ Good(Gd) → Excellent(Ex)이므로 각 1~5 값으로 매핑
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2}
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

datasets = label_encoder(datasets, qual_cols)     # train
test_sets = label_encoder(test_sets, qual_cols)   # test

#print(datasets.shape) # (1350, 14)
#print(test_sets.shape) # (1350, 13)

######################## 분류형 컬럼을 one hot encoding ########################
datasets = pd.get_dummies(datasets, columns = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
test_sets = pd.get_dummies(test_sets, columns = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])

######################## 분류형 컬럼을 one hot encoding ########################
# print(datasets.columns)
# print(test_sets.columns)
# print(datasets.shape) # (1350, 23)
# print(test_sets.shape) # (1350, 22)

####### x, y 분리 #######
x = datasets.drop(['target'], axis = 1)
y = datasets['target']

test_sets = test_sets.values         # pandas -->  numpy로 변경

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state = 66)
#print(x_train.shape, y_train.shape) # (1080, 22) (1080,)
#print(x_test.shape, y_test.shape) # (270, 22) (270,)

#scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler = QuantileTransformer()
#scaler = PowerTransformer(method = 'box-cox') # error
scaler = PowerTransformer(method = 'yeo-johnson') # 디폴트

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

########## 베이지안 옵티마이제이션 ##########
parms = {'max_depth': (3,7),                  # BayesianOptimization은 범위를 나타내는 것이므로 (3,7)이 아닌 3~7사이의 실수를 말하는 것!!!
          'learning_rate': (0.01,0.2),
          'n_estimators': (5000,10000),
          'min_child_weight': (0,3),
          'subsample': (0.5, 1),
          'colsample_bytree': (0.2,1),
          'reg_lambda': (0.001, 10),}

def xg_def(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, reg_lambda):
    xg_model = XGBRegressor(max_depth = int(max_depth),  # 베이지안은 실수로 받아들이기 때문에 max_depth의 경우 다시 int로 바꿔줘야한다.
                            learning_rate = learning_rate,
                            n_estimators = int(n_estimators),
                            min_child_weight = min_child_weight,
                            subsample = subsample,
                            colsample_bytree = colsample_bytree,
                            reg_lambda = reg_lambda)       
    xg_model.fit(x_train, y_train, eval_set = [(x_test, y_test)],
                 eval_metric = 'mae',
                 verbose = 1,
                 early_stopping_rounds = 50)
    y_predict = xg_model.predict(x_test)
    
    nmae = NMAE(y_test, y_predict)
    return nmae
    
bo = BayesianOptimization(f = xg_def, pbounds = parms, random_state = 66, verbose = 2)
bo.maximize(init_points = 10, n_iter = 200)  # init_points 초기 설정 값 //  n_iter 실제 훈련 값

print("=======bo.res========")
print(bo.res)
print("=======파라미터 튜닝 결과========")
#print(bo.max)    

target_list = []
for result in bo.res:
      target = result['target']
      target_list.append(target)
      
min_dict = bo.res[np.argmin(np.array(target_list))]
print(min_dict)

# {'target': 0.08905597988490348, 'params': {'colsample_bytree': 0.8819250794365909, 'learning_rate': 0.11901206948192891, 
#                                            'max_depth': 5.002515171899393, 'min_child_weight': 1.7166758886339086, 'n_estimators': 7421.505876139617, 
#                                            'reg_lambda': 8.355049478293875, 'subsample': 0.9054715831126499}}