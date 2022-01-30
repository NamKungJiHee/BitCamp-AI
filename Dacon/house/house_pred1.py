from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, PolynomialFeatures, QuantileTransformer
import warnings
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings(action='ignore')
import time
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor

#1. 데이터
path = "../_data/dacon/housing/"

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

#print(train.info())
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

le = LabelEncoder()
train['Exter Qual'] = le.fit_transform(train['Exter Qual'])
train['Kitchen Qual'] = le.fit_transform(train['Kitchen Qual'])
train['Bsmt Qual'] = le.fit_transform(train['Bsmt Qual'])
test['Exter Qual'] = le.fit_transform(test['Exter Qual'])
test['Kitchen Qual'] = le.fit_transform(test['Kitchen Qual'])
test['Bsmt Qual'] = le.fit_transform(test['Bsmt Qual'])

# def remove_outlier(input_data):
#     q1 = input_data.quantile(0.25) # 제 1사분위수
#     q3 = input_data.quantile(0.75) # 제 3사분위수
#     iqr = q3 - q1 # IQR(Interquartile range) 계산
#     minimum = q1 - (iqr * 1.5) # IQR 최솟값
#     maximum = q3 + (iqr * 1.5) # IQR 최댓값
#     ### IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치) ###
#     df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
#     return df_removed_outlier

# # # 이상치 제거한 데이터셋
# train = remove_outlier(train)
# test = remove_outlier(test)
# # #print(train[:40])

# # # 이상치 채워주기
# train = train.interpolate()
# test = test.interpolate()
#print(train[:40])
#print(test[:40])

# print(train.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.7)
# sns.heatmap(data=train.corr(), square=True, annot=True, cbar=True)
# plt.show()

#print(train.info())

train = train.drop(['id'], axis=1) 
test = test.drop(['id'], axis=1)

x = train.drop(['target'], axis = 1) 
y = train['target']

# # #print(x.shape, y.shape) # (1350, 11) (1350,)
# # #print(pd.Series(y).value_counts())
  
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test = scaler.transform(test)

#2. 모델링
# model = XGBRegressor(n_estimators = 4000, learning_rate =0.125, n_jobs = -1, 
#                      max_depth = 10,        
#                      min_child_weight = 1, 
#                      subsample = 1,        
#                      colsample_bytree = 1, 
#                      reg_alpha = 1,    
#                      reg_lambda = 0,)


# parameters = [
#     {'dt__max_depth' : [6, 8, 10], 'dt__min_samples_leaf' : [3, 5, 7]},
#     {'dt__min_samples_leaf' : [3, 5, 7], 'dt__min_samples_split' : [3, 5, 10]}]
# from sklearn.tree import DecisionTreeRegressor
# pipe = Pipeline([("ss",RobustScaler()), ("dt", RandomForestRegressor())])  
# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)
parameters = [{'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7,10]}]
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose=1, refit=True, n_jobs=-1)

#model = KNeighborsRegressor(n_neighbors=1000, leaf_size=40,n_jobs=-1)
#model = DecisionTreeRegressor(splitter="best", max_depth=1000, min_samples_split=4, min_samples_leaf=30)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

y_predict = model.predict(x_test)
print(NMAE(y_test, y_predict))

# 제출
result = model.predict(test)
print(result)
submit_file['target']= result
submit_file.to_csv(path + 'sample_submission.csv',index=False)