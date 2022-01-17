from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
#allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
#print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41


allAlgorithms = all_estimators(type_filter = 'regressor')  # regressor에 대한 모든 측정기

#print("allAlgorithms: ", allAlgorithms)  # [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수: 55

for (name, algorithm) in allAlgorithms:   # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),...
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        n_splits = 5
        kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)
        scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다. 
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률: ', acc, round(np.mean(scores),4))
    except:                     
        #continue   
        print(name, '은 에러')     
    
""" 
DecisionTreeRegressor 의 정답률:  0.011235955056179775 0.0172
DummyRegressor 은 에러
ElasticNet 은 에러
ElasticNetCV 은 에러
ExtraTreeRegressor 의 정답률:  0.011235955056179775 -0.0974
"""

