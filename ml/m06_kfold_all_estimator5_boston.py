from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_boston()
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
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률: ', r2, round(np.mean(scores),4))
    except:                     
        #continue   
        print(name, '은 에러')     
    
""" 
ARDRegression 의 정답률:  0.8119016106669674 0.6843
AdaBoostRegressor 의 정답률:  0.8957701483210189 0.7853
BaggingRegressor 의 정답률:  0.9144828861391499 0.8036
BayesianRidge 의 정답률:  0.8119880571377842 0.6841
CCA 의 정답률:  0.791347718442463 0.6351
DecisionTreeRegressor 의 정답률:  0.7755968489675353 0.7147
DummyRegressor 의 정답률:  -0.0005370164400797517 -0.0223
ElasticNet 의 정답률:  0.16201563080833714 0.1313
ElasticNetCV 의 정답률:  0.8113737663385278 0.6852
ExtraTreeRegressor 의 정답률:  0.7999216372533443 0.7065
ExtraTreesRegressor 의 정답률:  0.9329198548458454 0.8523
GammaRegressor 의 정답률:  0.1964792057029865 0.1684
GaussianProcessRegressor 의 정답률:  -1.578958693000398 -1.8099
GradientBoostingRegressor 의 정답률:  0.9440173735221447 0.855
HistGradientBoostingRegressor 의 정답률:  0.9323326124661162 0.8317
HuberRegressor 의 정답률:  0.7958373063951082 0.6666
IsotonicRegression 은 에러
KNeighborsRegressor 의 정답률:  0.8265307833211177 0.6519
KernelRidge 의 정답률:  0.803254958502079 0.5899
Lars 의 정답률:  0.7746736096721598 0.6651
LarsCV 의 정답률:  0.7981576314184016 0.6678
Lasso 의 정답률:  0.242592140544296 0.2115
LassoCV 의 정답률:  0.8125908596954046 0.6839
LassoLars 의 정답률:  -0.0005370164400797517 -0.0223
LassoLarsCV 의 정답률:  0.8127604328474283 0.684
LassoLarsIC 의 정답률:  0.8131423868817642 0.6756
LinearRegression 의 정답률:  0.8111288663608667 0.6822
LinearSVR 의 정답률:  0.7087073548465422 0.5683
MLPRegressor 의 정답률:  0.44130499786784216 0.1451
MultiOutputRegressor 은 에러
MultiTaskElasticNet 은 에러
MultiTaskElasticNetCV 은 에러
MultiTaskLasso 은 에러
MultiTaskLassoCV 은 에러
NuSVR 의 정답률:  0.6254681434531 0.515
OrthogonalMatchingPursuit 의 정답률:  0.5827617571381449 0.5181
OrthogonalMatchingPursuitCV 의 정답률:  0.78617447738729 0.6354
PLSCanonical 의 정답률:  -2.2317079741425734 -2.2912
PLSRegression 의 정답률:  0.8027313142007888 0.656
PassiveAggressiveRegressor 의 정답률:  0.7350726538914747 0.5893
PoissonRegressor 의 정답률:  0.67496007101481 0.5982
QuantileRegressor 의 정답률:  -0.020280478327147522 -0.0387
RANSACRegressor 의 정답률:  0.5885448854366226 0.4677
RadiusNeighborsRegressor 의 정답률:  0.41191760158788593 0.3148
RandomForestRegressor 의 정답률:  0.9259047412037633 0.8308
RegressorChain 은 에러
Ridge 의 정답률:  0.8087497007195745 0.6835
RidgeCV 의 정답률:  0.8116598578372426 0.6833
SGDRegressor 의 정답률:  0.8220219489421912 0.664
SVR 의 정답률:  0.6597910766772523 0.5398
StackingRegressor 은 에러
TheilSenRegressor 의 정답률:  0.7840350149366354 0.6479
TransformedTargetRegressor 의 정답률:  0.8111288663608667 0.6822
TweedieRegressor 의 정답률:  0.19473445117356525 0.1597
VotingRegressor 은 에러
"""

