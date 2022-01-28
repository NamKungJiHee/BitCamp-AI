from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures, QuantileTransformer
import warnings
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings(action='ignore')
import time
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = "../_data/dacon/housing/"

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

#print(train.info())
''' 
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   id              1350 non-null   int64
 1   Overall Qual    1350 non-null   int64
 2   Gr Liv Area     1350 non-null   int64
 3   Exter Qual      1350 non-null   object
 4   Garage Cars     1350 non-null   int64
 5   Garage Area     1350 non-null   int64
 6   Kitchen Qual    1350 non-null   object
 7   Total Bsmt SF   1350 non-null   int64
 8   1st Flr SF      1350 non-null   int64
 9   Bsmt Qual       1350 non-null   object
 10  Full Bath       1350 non-null   int64
 11  Year Built      1350 non-null   int64
 12  Year Remod/Add  1350 non-null   int64
 13  Garage Yr Blt   1350 non-null   int64
 14  target          1350 non-null   int64
dtypes: int64(12), object(3)
'''
le = LabelEncoder()
# le.fit(train['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
# train['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'] = le.transform(train['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])

# le.fit(test['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
# test['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'] = le.transform(test['Exter Qual', 'Kitchen Qual', 'Bsmt Qual'])
train['Exter Qual'] = le.fit_transform(train['Exter Qual'])
train['Kitchen Qual'] = le.fit_transform(train['Kitchen Qual'])
train['Bsmt Qual'] = le.fit_transform(train['Bsmt Qual'])
test['Exter Qual'] = le.fit_transform(test['Exter Qual'])
test['Kitchen Qual'] = le.fit_transform(test['Kitchen Qual'])
test['Bsmt Qual'] = le.fit_transform(test['Bsmt Qual'])

# print(train.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.7)
# sns.heatmap(data=train.corr(), square=True, annot=True, cbar=True)
# plt.show()

# print(train.info())
''' 
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   id              1350 non-null   int64
 1   Overall Qual    1350 non-null   int64
 2   Gr Liv Area     1350 non-null   int64
 3   Exter Qual      1350 non-null   int32
 4   Garage Cars     1350 non-null   int64
 5   Garage Area     1350 non-null   int64
 6   Kitchen Qual    1350 non-null   int32
 7   Total Bsmt SF   1350 non-null   int64
 8   1st Flr SF      1350 non-null   int64
 9   Bsmt Qual       1350 non-null   int32
 10  Full Bath       1350 non-null   int64
 11  Year Built      1350 non-null   int64
 12  Year Remod/Add  1350 non-null   int64
 13  Garage Yr Blt   1350 non-null   int64
 14  target          1350 non-null   int64
dtypes: int32(3), int64(12)
'''
train = train.drop(['id'], axis=1) 
test = test.drop(['id'], axis=1)

x = train.drop(['target'], axis = 1) 
y = train['target']

# #print(x.shape, y.shape) # (1350, 11) (1350,)
# #print(pd.Series(y).value_counts())
  
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

# 2. 모델링
model = XGBRegressor(n_estimators = 3000, learning_rate =0.15, n_jobs = -1, 
                     max_depth = 6,        
                     min_child_weight = 1, 
                     subsample = 1,        
                     colsample_bytree = 1, 
                     reg_alpha = 1,    
                     reg_lambda = 0,)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

y_pred = model.predict(x_test)

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
print(NMAE(y_test, y_pred))

# 제출
result = model.predict(test)
submit_file['target']= result
submit_file.to_csv(path + 'sample_submission.csv',index=False)