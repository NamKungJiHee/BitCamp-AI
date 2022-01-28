import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import os
import sys
import platform
import random
import math
from typing import List, Dict,Tuple 
import sklearn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier
SEED=47

path = "../_data/dacon/Jobcare/"

train_data=pd.read_csv(path+'train.csv')

test_data=pd.read_csv(path+'test.csv')

code_d=pd.read_csv(path+'속성_D_코드.csv').iloc[:,:-1]

code_h=pd.read_csv(path+'속성_H_코드.csv')

code_l=pd.read_csv(path+'속성_L_코드.csv')

submit_file = pd.read_csv(path + "sample_submission.csv")

#print(train_data.shape, test_data.shape)  (501951, 35) (46404, 34)

#code_h.info()

print(train_data.info)

for col1 in train_data.columns:
    n_nan1 = train_data[col1].isnull().sum()
    if n_nan1>0:
      msg1 = '{:^20}에서 결측치 개수: {}개'.format(col1,n_nan1)
      print(msg1)
    else:
        print('결측치가 없습니다.')

for col2 in train_data.columns:
    n_nan2 = train_data[col2].isnull().sum()
    if n_nan2>0:
        msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
        print(msg2)
    else:
      print('결측치가 없습니다.')


code_d.columns= ["attribute_d_d","attribute_d_s","attribute_d_m","attribute_d_l"]
code_h.columns= ["attribute","attribute_h","attribute_h_p"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]


def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)
  
def preprocess_data(
                    df:pd.DataFrame,is_train:bool = True, cols_merge:List[Tuple[str,pd.DataFrame]] = []  , cols_equi:List[Tuple[str,str]]= [] ,
                    cols_drop:List[str] = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt"]
                    )->Tuple[pd.DataFrame,np.ndarray]:
    df = df.copy()

    y_data = None
    if is_train:
        y_data = df["target"].to_numpy()
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df,df_code,col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2] ).astype(int)

    df = df.drop(columns=cols_drop)
    return (df , y_data)

cols_merge = [
              ("person_prefer_d_1" , code_d),
              ("person_prefer_d_2" , code_d),
              ("person_prefer_d_3" , code_d),
              ("contents_attribute_d" , code_d),
              ("person_prefer_h_1" , code_h),
              ("person_prefer_h_2" , code_h),
              ("person_prefer_h_3" , code_h),
              ("contents_attribute_h" , code_h),
              ("contents_attribute_l" , code_l),
]

cols_equi = [

    ("contents_attribute_c","person_prefer_c"),
    ("contents_attribute_e","person_prefer_e"),

    ("person_prefer_d_2_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_2_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_2_attribute_d_l" , "contents_attribute_d_attribute_d_l"),
    ("person_prefer_d_3_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_3_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_3_attribute_d_l" , "contents_attribute_d_attribute_d_l"),

    ("person_prefer_h_1_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_2_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_3_attribute_h_p" , "contents_attribute_h_attribute_h_p"),

]


cols_drop = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", ]

x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
x_train.shape , y_train.shape , x_test.shape

cat_features = x_train.columns[x_train.nunique() > 2].tolist()

random_state= 69
is_holdout = False
n_splits = 10
iterations = 5000
patience = 50

cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

scores = []
models = []

models = []
for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []

    model = CatBoostClassifier(iterations=iterations,random_state=SEED,task_type="GPU",eval_metric="F1",cat_features=cat_features,one_hot_max_size=4)
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=patience ,
            verbose = 100
        )
    
    models.append(model)
    scores.append(model.get_best_score()["validation"]["F1"])
    if is_holdout:
        break    

print(scores)
print(np.mean(scores))

threshold = 0.35

pred_list = []
scores = []
for i,(tri, vai) in enumerate( cv.split(x_train) ):
    pred = models[i].predict_proba(x_train.iloc[vai])[:, 1]
    pred = np.where(pred >= threshold , 1, 0)
    score = f1_score(y_train[vai],pred)
    scores.append(score)
    pred = models[i].predict_proba(x_test)[:, 1]
    pred_list.append(pred)
    
print(scores)
print(np.mean(scores))

pred = np.mean( pred_list , axis = 0 )
pred = np.where(pred >= threshold , 1, 0)

submit_file['target']=pred

submit_file.to_csv(path + 'sample_submission.csv',index=False)