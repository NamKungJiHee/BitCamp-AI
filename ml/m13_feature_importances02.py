# 중요도 떨어지는 컬럼 제거 후 13-1의 파일 결과값과 비교!

import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target
#print(datasets.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = np.delete(x,[0,1],axis=1)  ######################넘파이에서 컬럼 삭제######################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#model = DecisionTreeClassifier(max_depth=5, random_state=66)
#model = RandomForestClassifier()
model = XGBClassifier()

#3) 훈련

model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 

#print("RandomForestClassifier: ", result)
print("accuracy: ", acc)

print(model.feature_importances_)

''' 
accuracy:  0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]

-----------컬럼 첫번째꺼 삭제 후--------------
accuracy:  0.9666666666666667
[0.0125026  0.53835801 0.44913938]   ----> 불필요한 컬럼 제거함으로써 연산량 줄이기!

-----------컬럼 두번째꺼 삭제 후--------------
accuracy:  0.9333333333333333
[0.54517411 0.45482589]
'''
