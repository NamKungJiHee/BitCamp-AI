from random import random
import numpy as np, pandas as pd
import time
from datetime import datetime
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import autokeras as ak

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])  
                                                                       
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

###################### y값 로그변환 ######################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##########################################################

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

ak_model = ak.StructuredDataRegressor(overwrite=True, max_trials= 2, loss = 'mean_absolute_error')    # 수치는 보통 StructuredDataRegressor을 쓴다.

start = time.time()
ak_model.fit(x_train, y_train, epochs = 5) #, epochs = 1000, validation_split = 0.2)  # epoch default = 1000 
end = time.time() - start

model = ak_model.export_model() # 가장 좋은 모델을 뽑겠다.
#save model
# autokeras.com에서 참고

y_predict = model.predict(x_test) # 가장 좋은 모델로

results = model.evaluate(x_test, y_test)
print("results: ", np.round(results,6))

#print(y_test.shape, y_predict.shape) # (270,) (270, 1)
y_predict = y_predict.reshape(270,)

nmae = NMAE(np.expm1(y_test), np.expm1(y_predict))    # 로그를 다시 지수변환해주는 것!!!
print("NMAE: ", np.round(nmae, 6))
''' 
results:  [0.94497  1.615946]
NMAE:  1.528384
'''
############################# 제출용 #############################
colsample_bytree= 0.8819
learning_rate = 0.119
max_depth= 5
min_child_weight = 1.7166
n_estimators = 7421
reg_lambda= 8.355
subsample = 0.9054

y_submit = model.predict(test_sets)

y_submit = np.expm1(y_submit)
submit_sets.target = y_submit

path_save_csv = './hackathon/dacon/housing/_save_csv/'
now1 = datetime.now()
now_date = now1.strftime("%m%d_%H%M") #연도와 초를 빼본다.

submit_sets.to_csv(path_save_csv + now_date + '_' + str(round(nmae,4)) + '.csv')

#model.summary()

with open(path_save_csv + now_date + '_' + str(round(nmae,4)) + 'submit.txt', 'a') as file: # 'a'안하면 파일 각각 개별로 생성
    file.write("\n=======================")
    file.write('저장시간: ' + now_date + '\n')
    file.write('scaler: ' + str(scaler) + '\n')
    file.write('colsample_bytree: ' + str('colsample_bytree') + '\n')
    file.write('learning_rate: ' + str('learning_rate') + '\n')
    file.write('max_depth: ' + str('max_depth') + '\n')
    file.write('min_child_weight: ' + str('min_child_weight') + '\n')
    file.write('n_estimators: ' + str('n_estimators') + '\n')
    file.write('reg_lambda: ' + str('reg_lambda') + '\n')
    file.write('subsample: ' + str('subsample') + '\n')
    
    file.write('걸린시간: ' + str(round(end,4)) + '\n')
    file.write('evaluate: ' + str(np.round(results,6)) + '\n')
    file.write('NMAE: ' + str(round(nmae,4)) + '\n')
    # file.close()   앞에 with를 써주면 close 안해줘도 된다.