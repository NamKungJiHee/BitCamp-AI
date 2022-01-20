import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action = "ignore")
import sklearn as sk

#1. 데이터
#datasets = load_boston()
datasets = load_breast_cancer()
#datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> (20640, 8)  # boston
print(x.shape) # (569, 30) # cancer

pca = PCA(n_components = 25) # 컬럼 13개를 8개로 줄여주겠다. # 차원 축소라는 것은 y를 건들이는 것이 아니다.(x만 건들인다)
x = pca.fit_transform(x)
#print(x)
#print(x.shape) # (506, 5) 
print(x.shape) # (569, 25)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle = True)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric= 'error')  
#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과: ", results)

''' 차원 축소 기법
# boston #
(506, 13)
결과:  0.9221188601856797
===========================
(506, 8)
결과:  0.7856968255504542

# cancer #
(569, 30)
결과:  0.9736842105263158
(569, 25)
결과:  0.9649122807017544
'''
# print(sk.__version__) # 1.0.1