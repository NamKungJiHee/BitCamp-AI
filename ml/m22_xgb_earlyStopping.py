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

#2.  모델
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

''' 
[229]   validation_0-rmse:1.26583       validation_1-rmse:2.14090
[230]   validation_0-rmse:1.26132       validation_1-rmse:2.14157
[231]   validation_0-rmse:1.25891       validation_1-rmse:2.14182
[232]   validation_0-rmse:1.25497       validation_1-rmse:2.14286
[233]   validation_0-rmse:1.25249       validation_1-rmse:2.14205
[234]   validation_0-rmse:1.25073       validation_1-rmse:2.14286
[235]   validation_0-rmse:1.24711       validation_1-rmse:2.14093
[236]   validation_0-rmse:1.24525       validation_1-rmse:2.14107
[237]   validation_0-rmse:1.24249       validation_1-rmse:2.14111
[238]   validation_0-rmse:1.23946       validation_1-rmse:2.14141
[239]   validation_0-rmse:1.23652       validation_1-rmse:2.14179
걸린시간:  0.3650238513946533
results:  0.9452
r2:  0.9452
'''
print("=========================")
hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist['validation_0']['rmse'], marker=".", c='red', label='test_set')
#plt.plot(hist['validation_1']['rmse'], marker='.', c='blue', label='test_set')
plt.grid() 
plt.title('loss_rmse')
plt.ylabel('loss_rmse')
plt.xlabel('epoch')
plt.legend(loc='upper right') 
plt.show()