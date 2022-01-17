from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_breast_cancer()
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
    except:                     # 에러나는 것 빼고 계속해라.
        #continue   
        print(name, '은 에러')     
    
""" 
AdaBoostClassifier 의 정답률:  0.9473684210526315 0.9604
BaggingClassifier 의 정답률:  0.956140350877193 0.9429
BernoulliNB 의 정답률:  0.6403508771929824 0.611
CalibratedClassifierCV 의 정답률:  0.9649122807017544 0.9692
CategoricalNB 은 에러
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.7807017543859649 0.8462
DecisionTreeClassifier 의 정답률:  0.9298245614035088 0.9297
DummyClassifier 의 정답률:  0.6403508771929824 0.6242
ExtraTreeClassifier 의 정답률:  0.9298245614035088 0.8945
ExtraTreesClassifier 의 정답률:  0.9649122807017544 0.9692
GaussianNB 의 정답률:  0.9210526315789473 0.9319
GaussianProcessClassifier 의 정답률:  0.9649122807017544 0.9604
GradientBoostingClassifier 의 정답률:  0.956140350877193 0.9648
HistGradientBoostingClassifier 의 정답률:  0.9736842105263158 0.9648
KNeighborsClassifier 의 정답률:  0.956140350877193 0.9692
LabelPropagation 의 정답률:  0.9473684210526315 0.9714
LabelSpreading 의 정답률:  0.9473684210526315 0.9692
LinearDiscriminantAnalysis 의 정답률:  0.9473684210526315 0.9473
LinearSVC 의 정답률:  0.9736842105263158 0.9802
LogisticRegression 의 정답률:  0.9649122807017544 0.9582
LogisticRegressionCV 의 정답률:  0.9736842105263158 0.9714
MLPClassifier 의 정답률:  0.9824561403508771 0.967
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.8508771929824561 0.8462
NearestCentroid 의 정답률:  0.9298245614035088 0.9341
NuSVC 의 정답률:  0.9473684210526315 0.9451
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9122807017543859 0.9648
Perceptron 의 정답률:  0.9736842105263158 0.967
QuadraticDiscriminantAnalysis 의 정답률:  0.9385964912280702 0.9516
RadiusNeighborsClassifier 은 에러
RandomForestClassifier 의 정답률:  0.9649122807017544 0.956
RidgeClassifier 의 정답률:  0.9473684210526315 0.9538
RidgeClassifierCV 의 정답률:  0.9473684210526315 0.9604
SGDClassifier 의 정답률:  0.9736842105263158 0.9692
SVC 의 정답률:  0.9736842105263158 0.9736
StackingClassifier 은 에러
VotingClassifier 은 에러
"""

