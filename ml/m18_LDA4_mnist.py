import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')
from tensorflow.keras.datasets import mnist
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

#print(x_test.shape) # (10000, 784)
#print("LDA전: ", x_train.shape) # LDA전:  (60000, 784)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
#print("LDA후: ", x_train.shape)  # LDA후:  (60000, 9)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
start = time.time()
# model.fit(x_train, y_train, eval_metric='error')  # 이진분류
#model.fit(x_train, y_train, eval_metric='merror')  # 다중분류
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print("results: ", results)
print("걸린시간: ", end - start)

""" 
1. mnist
LDA전:  (60000, 784)
LDA후:  (60000, 9)
results:  0.9163
걸린시간:  32.19190859794617
"""