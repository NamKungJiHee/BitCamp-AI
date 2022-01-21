# n_component > 0.95 이상   '154'
# xgboost, gridSearch 또는 RandomSearch를 쓸 것
# m16결과를 뛰어넘어라

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x = np.append(x_train, x_test, axis = 0) # train과 test를 행(axis = 0)으로 합치겠다 
#print(x.shape) # (70000, 28, 28) 
#x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
#print(x.shape) # (70000, 784)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

#print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

pca = PCA(n_components=154)
# x = pca.fit_transform(x)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

parameters = [
    {"xg__n_estimators":[100,200,300], "xg__learning_rate":[0.1,0.3,0.001,0.01],"xg__max_depth":[4,5,6]},
    {"xg__n_estimators":[90,100,110], "xg__learning_rate":[0.1, 0.001, 0.01],"xg__max_depth":[4,5,6], "xg__colsample_bytree":[0.6, 0.9, 1]},
    {"xg__n_estimators":[90,110], "xg__learning_rate":[0.1, 0.001, 0.5],"xg__max_depth":[4,5,6], "xg__colsample_bytree":[0.6, 0.9, 1],"xg__colsample_bylevel":[0.6,0.7,0.9]}]

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline

pipe = Pipeline([('mm',MinMaxScaler()),('xg', XGBClassifier(eval_metric='merror'))])
model = GridSearchCV(pipe, parameters, cv = 2, verbose = 1, n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train,eval_metric='merror')
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)
from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', acc)
print('걸린 시간 :', end - start)

''' 
GridSearchCV


'''