import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
''' 
* GridSearchCV 란?
교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에 적용할 수 있게 해주는 사이킷런의 기능 중 하나
====================================================
max_depth 5,6,7  (트리의 깊이)
learning_rate 0.1 , 0.001, 0.0001  
(= optimizer, 최적의 loss값을 구하려고 )
( 너무 크면 튕겨나가고 너무 작으면 소실의 문제)
n_estimate 100,500,1000
'''
#1) 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#2) 모델구성
model = SVC(C=1, kernel = 'linear', degree=3)
# model = GridSearchCV(SVC(), parameters, cv = kfold, verbose=1, 
#                      refit=True)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                                                # Fitting 5 folds for each of 42 candidates, totalling 210 fits
                                                                # refit = True : 가장 좋은 값을 뽑아내겠다.
                                                                
#3) 훈련
model.fit(x_train, y_train)

#4) 평가, 예측
print("model.score: ", model.score(x_test, y_test)) # model.score:  0.9666666666666667  = model.evaluate과 같다
 
y_predict = model.predict(x_test)
print("accuracy_score: ", accuracy_score(y_test, y_predict))  # accuracy_score:  0.9666666666666667
