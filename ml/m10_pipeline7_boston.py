import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1) 데이터
datasets = load_boston()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2) 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

#model = SVC()
model = make_pipeline(MinMaxScaler(), RandomForestRegressor()) 

#3) 훈련
model.fit(x_train, y_train) # 훈련시켜줌

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 

print("model.score: ", result)

""" 
pipeline 사용
model.score:  0.921477466108204

HalvingGridSearch 사용
accuracy_score:  0.9222713364082966
"""