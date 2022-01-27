from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score, r2_score, accuracy_score
import time
### Pandas ###
#1. 데이터 
path = '../_data/winequality/'
# datasets = pd.read_csv(path +"winequality-white.csv", header = 0, sep=';')
#print(datasets.shape) #(4898, 12)
# x = datasets.drop(['quality','chlorides','pH'], axis = 1)
# y = datasets['quality']

# for i in range(len(y)):
#     if y[i] < 4 : 
#         y[i] = 4
#     elif y[i] > 8:
#         y[i] = 8
# print(np.unique(y))  # [4 5 6 7 8]
# print("라벨: ", np.unique(y, return_counts=True))
# 라벨:  (array([4, 5, 6, 7, 8], dtype=int64), array([ 183, 1457, 2198,  880,  180], dtype=int64))
########################################################################################################
### Numpy ###
#1. 데이터
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') 
datasets = datasets.values #  pandas --> numpy로 바꿔주기

x = datasets[:, :11]  
y = datasets[:, 11] 
#print(y.shape) # (4898,)

newlist = []
for i in y: 
    #print(i)
    if i <=4:                   
        newlist += [0] 
    elif i <=7:                
        newlist += [1]
    else:                      
        newlist += [2]
                    
y = np.array(newlist)

print(np.unique(y, return_counts = True)) # (array([0, 1, 2]), array([183, 4535,  180], dtype=int64))
        
x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8, stratify = y)
#print(x_train.shape, y_train.shape)

scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=1000,
    learning_rate = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample =1,
    colsample_bytree =1,
    reg_alpha =1,              
    reg_lambda=0,              
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor')

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',          
          early_stopping_rounds=20)
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
걸린시간 : 5.479347467422485
results : 0.9377551020408164
acc : 0.9377551020408164
f1_score : 0.9377551020408164
'''