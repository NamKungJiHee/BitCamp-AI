from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_wine()
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
AdaBoostClassifier 의 정답률:  0.8888888888888888 0.8099
BaggingClassifier 의 정답률:  1.0 0.9155
BernoulliNB 의 정답률:  0.4166666666666667 0.3175
CalibratedClassifierCV 의 정답률:  0.9722222222222222 0.9786
CategoricalNB 의 정답률:  0.5 nan
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.8611111111111112 0.8648
DecisionTreeClassifier 의 정답률:  0.9722222222222222 0.8727
DummyClassifier 의 정답률:  0.4166666666666667 0.3958
ExtraTreeClassifier 의 정답률:  0.9166666666666666 0.8512
ExtraTreesClassifier 의 정답률:  1.0 0.9643
GaussianNB 의 정답률:  1.0 0.9786
GaussianProcessClassifier 의 정답률:  1.0 0.9643
GradientBoostingClassifier 의 정답률:  0.9722222222222222 0.88
HistGradientBoostingClassifier 의 정답률:  0.9722222222222222 0.9507
KNeighborsClassifier 의 정답률:  1.0 0.9571
LabelPropagation 의 정답률:  1.0 0.9645
LabelSpreading 의 정답률:  1.0 0.9645
LinearDiscriminantAnalysis 의 정답률:  1.0 0.9857
LinearSVC 의 정답률:  0.9722222222222222 0.9786
LogisticRegression 의 정답률:  1.0 0.9643
LogisticRegressionCV 의 정답률:  0.9722222222222222 0.9643
MLPClassifier 의 정답률:  0.9722222222222222 0.9714
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.9444444444444444 0.8805
NearestCentroid 의 정답률:  1.0 0.9574
NuSVC 의 정답률:  1.0 0.9714
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9722222222222222 0.9929
Perceptron 의 정답률:  0.9722222222222222 0.9788
QuadraticDiscriminantAnalysis 의 정답률:  0.9722222222222222 0.9788
RadiusNeighborsClassifier 의 정답률:  0.9722222222222222 0.9222
RandomForestClassifier 의 정답률:  1.0 0.9717
RidgeClassifier 의 정답률:  1.0 0.9786
RidgeClassifierCV 의 정답률:  0.9722222222222222 0.9786
SGDClassifier 의 정답률:  0.9722222222222222 0.9714
SVC 의 정답률:  1.0 0.9786
StackingClassifier 은 에러
VotingClassifier 은 에러
"""

