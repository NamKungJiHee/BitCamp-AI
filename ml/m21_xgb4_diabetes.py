from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
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
datasets = load_diabetes()
x = datasets.data
y = datasets['target']
#print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.  모델
#model = XGBRegressor()                  ###### n_estimators : tensorflow에서 epochs값과 같다. ######
model = XGBRegressor(n_estimators = 2000, learning_rate =0.025, n_jobs = -1, 
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
          eval_metric= 'rmse', )
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", round(r2,4))

''' 
걸린시간:  1.1287841796875
results:  0.3415
r2:  0.3415
'''
print("=========================")
hist = model.evals_result()
print(hist)

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