from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = fetch_covtype()
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
allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
#print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41


#allAlgorithms = all_estimators(type_filter = 'regressor')  # regressor에 대한 모든 측정기

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
AdaBoostClassifier 의 정답률:  0.5028613719095032 0.5124
BaggingClassifier 의 정답률:  0.9627548342125418 0.9569
BernoulliNB 의 정답률:  0.631833945767321 0.6323
CalibratedClassifierCV 의 정답률:  0.7122621619063191 0.713
CategoricalNB 의 정답률:  0.6321437484402296 nan
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.6225742880992745 0.6216
DecisionTreeClassifier 의 정답률:  0.9399929433835615 0.9321
DummyClassifier 의 정답률:  0.48625250638968015 0.4879
ExtraTreeClassifier 의 정답률:  0.8554082080497061 0.8546
ExtraTreesClassifier 의 정답률:  0.954304105745979 0.9489
GaussianNB 의 정답률:  0.09079800005163378 0.0907
GaussianProcessClassifier 은 에러
"""

