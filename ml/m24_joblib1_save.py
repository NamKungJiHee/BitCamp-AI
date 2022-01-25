from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터
#datasets = fetch_california_housing()
datasets = load_boston()

x = datasets.data
y = datasets['target']
#print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
#model = XGBRegressor()                 
model = XGBRegressor(n_estimators = 2000, learning_rate =0.025, n_jobs = -1, 
                     max_depth = 4,         
                     min_child_weight = 1,  
                     subsample = 1,      
                     colsample_bytree = 1,  
                     reg_alpha = 1,    
                     reg_lambda = 0,)  
                                                             
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1,
          eval_set = [(x_test, y_test)],  
          eval_metric= 'rmse',
          early_stopping_rounds= 50    # 여기에서는 callback안하고 그냥 early_stopping_rounds만 써줘도 된다! 
          )  
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", round(r2,4))

print("=========================")
hist = model.evals_result()
print(hist)

# 저장
# import pickle
path = './_save/'
# pickle.dump(model, open(path +'m23_pickle1_save.dat', 'wb')) # write   

import joblib
joblib.dump(model, path + 'm24_joblib1_save.dat')

''' 
results:  0.9453
r2:  0.9453
'''