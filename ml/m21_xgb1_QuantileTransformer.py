from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
##################################
# QuantileTransformer: 
# 1000개의 quantile을 이용해서 데이터를 균등하게 분포시킨다. quantile을 이용하는 RobustScaler와 똑같이 이상치에 민감하지 않다. 
# 전체 데이터를 MinMaxScaler와 비슷하게 0과 1사이로 압축시킨다.
# PolynomialFeatures:
# 입력값 x를 다항식으로 변환시킨다.
# PowerTransformer:
# 데이터의 특성별로 정규분포형태에 가깝도록 변환시킨다.
##################################

#1. 데이터
#datasets = fetch_california_housing()
datasets = load_boston()

x = datasets.data
y = datasets['target']
#print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) #, stratify = y)

#scaler = MinMaxScaler()
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.  모델
#model = XGBRegressor()                  ###### n_estimators : tensorflow에서 epochs값과 같다. ######
model = XGBRegressor(n_estimators = 2500, learning_rate =0.15, n_jobs = -1) # verbose = 1 ) 
                                                             
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1)
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

''' 
## fetch_california_housing ##
걸린시간:  19.31609296798706
results:  0.8566291699938181
r2:  0.8566291699938181

## load_boston ##  
걸린시간:  1.0371181964874268
results:  0.9385790281575488
r2:  0.9385790281575488
'''
print("=========================")
# hist = model.evals_result()
# print(hist)
