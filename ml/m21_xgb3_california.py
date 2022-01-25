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
datasets = fetch_california_housing()
#datasets = load_boston()

x = datasets.data
y = datasets['target']
#print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.  모델
#model = XGBRegressor()                  ###### n_estimators : tensorflow에서 epochs값과 같다. ######
model = XGBRegressor(n_estimators = 200, learning_rate =0.1, n_jobs = -1, 
                     max_depth = 4,         # max_depth = 소수점 x
                     min_child_weight = 1,  # 가중치 합의 최소 / 값이 높을수록 과적합 방지
                     subsample = 1,         # 학습에 사용하는 데이터 샘플링 비율(값이 낮을수록 과적합 방지)
                     colsample_bytree = 1,  # 각 tree별 사용된 feature의 비율
                     reg_alpha = 1,    # (가중치) 규제 L1 
                     reg_lambda = 0,)  # 규제 L2
                                                             
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1,
          eval_set = [(x_train, y_train),(x_test, y_test)],   # eval_set: 평가를 x,ytest로 할거다./ 훈련 과정 보려면 x_train, y_train 넣기
          eval_metric= 'rmse',             # rmse, mae, logloss, error
          )  
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", round(r2,4))

''' 
# XGB default로 돌렸을 때
걸린시간:  1.0476789474487305
results:  0.8434
r2:  0.8434
==============================
# XGB 튠했을 때
걸린시간:  25.80327582359314
results:  0.8628
r2:  0.8628
=============================
[1998]  validation_0-rmse:0.43872
[1999]  validation_0-rmse:0.43871   회귀모델: rmse
'''
print("=========================")
hist = model.evals_result()
print(hist)
''' 
=========================
{'validation_0': OrderedDict([('rmse', [1.898571, 1.858498, 1.819566, 1.781689, 1.744898, 1.709089, 1.674333, 1.640396, 1.607613, 1.575693])]), 
'validation_1': OrderedDict([('rmse', [1.934801, 1.893832, 1.854026, 1.815243, 1.777547, 1.740867, 1.705343, 1.670748, 1.637181, 1.604616])])}

# validation_0 :  eval_set의 x,y_train
# validation_1 :  eval_set의 x,y_test
'''
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist['validation_0']['rmse'], marker=".", c='red', label='train_set')
plt.plot(hist['validation_1']['rmse'], marker='.', c='blue', label='test_set')
plt.grid() 
plt.title('loss_rmse')
plt.ylabel('loss_rmse')
plt.xlabel('epoch')
plt.legend(loc='upper right') 
plt.show()