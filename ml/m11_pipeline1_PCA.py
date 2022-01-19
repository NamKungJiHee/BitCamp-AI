import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

''' 
pipeline은 계속적으로 엮어나갈 수 있다.  
'''

#1) 데이터
datasets = load_iris()
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA  # 주성분분석 (고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법)
                                       
#model = SVC()
#model = make_pipeline(MinMaxScaler(), StandardScaler(), SVC()) ####### Scaler 2번 써도 가능 #######
model = make_pipeline(MinMaxScaler(), PCA(), SVC()) # PCA도 fit,transform이 된다!

#3) 훈련
model.fit(x_train, y_train) # 훈련시켜줌

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 

print("model.score: ", result)

""" 
model.score:  1.0
"""