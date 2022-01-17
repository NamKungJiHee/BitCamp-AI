from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_iris()
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
        print(name, '의 정답률: ', acc, scores, round(np.mean(scores),4))
    except:                     # 에러나는 것 빼고 계속해라.
        #continue   
        print(name, '은 에러')
        
        

        
    
""" 
AdaBoostClassifier 의 정답률:  0.6333333333333333 [0.95833333 0.75       0.91666667 0.83333333 0.875     ] 0.8667
BaggingClassifier 의 정답률:  0.9333333333333333 [0.95833333 0.95833333 0.95833333 1.         0.875     ] 0.95
BernoulliNB 의 정답률:  0.4 [0.29166667 0.25       0.25       0.45833333 0.375     ] 0.325
CalibratedClassifierCV 의 정답률:  0.9666666666666667 [0.95833333 0.875      0.79166667 0.875      0.83333333] 0.8667
CategoricalNB 의 정답률:  0.3333333333333333 [       nan        nan 0.375             nan 0.29166667] nan
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.6666666666666666 [0.75       0.66666667 0.58333333 0.625      0.70833333] 0.6667
DecisionTreeClassifier 의 정답률:  0.9666666666666667 [0.95833333 0.95833333 0.95833333 1.         0.875     ] 0.95
DummyClassifier 의 정답률:  0.3 [0.25       0.25       0.20833333 0.25       0.33333333] 0.2583
ExtraTreeClassifier 의 정답률:  0.9 [0.91666667 0.95833333 0.91666667 1.         0.83333333] 0.925
ExtraTreesClassifier 의 정답률:  0.9333333333333333 [0.95833333 0.95833333 0.95833333 1.         0.875     ] 0.95
GaussianNB 의 정답률:  0.9666666666666667 [0.95833333 0.95833333 0.91666667 1.         0.875     ] 0.9417
GaussianProcessClassifier 의 정답률:  0.9666666666666667 [0.95833333 0.91666667 0.83333333 0.95833333 0.79166667] 0.8917
GradientBoostingClassifier 의 정답률:  0.9333333333333333 [0.95833333 0.91666667 0.91666667 1.         0.875     ] 0.9333
HistGradientBoostingClassifier 의 정답률:  0.8666666666666667 [0.91666667 0.95833333 0.95833333 1.         0.875     ] 0.9417
KNeighborsClassifier 의 정답률:  1.0 [1.         0.95833333 0.95833333 1.         0.91666667] 0.9667
LabelPropagation 의 정답률:  0.9666666666666667 [0.95833333 0.95833333 0.95833333 1.         0.91666667] 0.9583
LabelSpreading 의 정답률:  0.9666666666666667 [0.95833333 0.95833333 0.95833333 1.         0.875     ] 0.95
LinearDiscriminantAnalysis 의 정답률:  1.0 [1.         0.95833333 0.95833333 1.         0.95833333] 0.975
LinearSVC 의 정답률:  0.9666666666666667 [0.95833333 0.91666667 0.83333333 1.         0.875     ] 0.9167
LogisticRegression 의 정답률:  0.9666666666666667 [0.95833333 0.91666667 0.83333333 0.95833333 0.79166667] 0.8917
LogisticRegressionCV 의 정답률:  1.0 [0.95833333 0.95833333 0.95833333 1.         0.91666667] 0.9583
MLPClassifier 의 정답률:  0.9333333333333333 [0.91666667 1.         0.83333333 0.95833333 0.91666667] 0.925
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.6333333333333333 [0.58333333 0.66666667 0.58333333 0.625      0.625     ] 0.6167
NearestCentroid 의 정답률:  0.9666666666666667 [0.95833333 0.91666667 0.91666667 0.95833333 0.79166667] 0.9083
NuSVC 의 정답률:  0.9666666666666667 [0.95833333 0.95833333 0.95833333 1.         0.91666667] 0.9583
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9333333333333333 [0.875      0.91666667 0.95833333 0.75       0.83333333] 0.8667
Perceptron 의 정답률:  0.9333333333333333 [0.75       0.66666667 0.95833333 0.79166667 0.875     ] 0.8083
QuadraticDiscriminantAnalysis 의 정답률:  1.0 [0.91666667 0.95833333 0.95833333 1.         0.95833333] 0.9583
RadiusNeighborsClassifier 의 정답률:  0.4666666666666667 [0.45833333 0.5        0.58333333 0.375      0.45833333] 0.475
RandomForestClassifier 의 정답률:  0.9 [0.95833333 0.91666667 0.95833333 1.         0.875     ] 0.9417
RidgeClassifier 의 정답률:  0.9333333333333333 [0.91666667 0.875      0.75       0.79166667 0.79166667] 0.825
RidgeClassifierCV 의 정답률:  0.8333333333333334 [0.91666667 0.83333333 0.83333333 0.83333333 0.79166667] 0.8417
SGDClassifier 의 정답률:  0.9333333333333333 [0.91666667 0.79166667 1.         1.         0.70833333] 0.8833
SVC 의 정답률:  1.0 [0.95833333 0.95833333 0.95833333 1.         0.91666667] 0.9583
StackingClassifier 은 에러
VotingClassifier 은 에러
"""

