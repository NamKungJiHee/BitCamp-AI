import numpy as np
import time
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

print("LDA 전 : ", x_train.shape)  # LDA 전 :  (60000, 784)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA후: ", x_train.shape) # LDA후:  (60000, 9)

parameters = [
    {"n_estimators":[90], "learning_rate":[0.1, 0.5],"max_depth":[6], "colsample_bytree":[0.9, 1],"colsample_bylevel":[0.6,0.9]}]

#2. 모델
#from sklearn.pipeline import make_pipeline, Pipeline

model = GridSearchCV(XGBClassifier(), parameters, cv=5, verbose=1, refit=True, n_jobs = -1)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, eval_metric= 'merror')
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print("results: ", results)
print("걸린시간: ", end - start)
''' 
LDA 전 :  (60000, 784)
LDA후:  (60000, 9)
results:  0.9157
걸린시간:  419.69882106781006
'''