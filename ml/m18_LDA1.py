from codecs import ignore_errors
from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

''' 
PCA와 LDA 모두 데이터셋의 차원 개수를 줄이는 선형 변환 기법이지만 PCA는 비지도, LDA는 지도학습이라는 점에서 다름
'''
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
#print(x.shape)   # (569, 30)

# pca = PCA(n_components=20)
# x = pca.fit_transform(x)  # (569, 20)

lda = LinearDiscriminantAnalysis()      # (n_components=20) error
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1)
x = lda.fit_transform(x, y)
print(x.shape)  # (569, 1) # label이 한개로 줄어들었다

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBRegressor()
#model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric='error') # 이진분류일떄 사용!!!!!

#4. 평가, 예측
results = model.score(x_test, y_test)
print("results : ", results)

''' 
LinearDiscriminantAnalysis 사용시
(569, 1)
results :  0.9824561403508771
'''