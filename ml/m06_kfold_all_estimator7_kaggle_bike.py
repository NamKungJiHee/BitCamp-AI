from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score,  r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

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
ARDRegression 의 정답률:  0.24926480107309645 0.2606
AdaBoostRegressor 의 정답률:  0.2038386403555642 0.225
BaggingRegressor 의 정답률:  0.20091776101087977 0.2465
BayesianRidge 의 정답률:  0.24957561720517973 0.2604
CCA 의 정답률:  -0.1875185306363374 -0.12
DecisionTreeRegressor 의 정답률:  -0.19906081133471565 -0.199
DummyRegressor 의 정답률:  -0.0006494197429334214 -0.0011    
ElasticNet 의 정답률:  0.060591067855856995 0.0609
ElasticNetCV 의 정답률:  0.24125986412862332 0.2494
ExtraTreeRegressor 의 정답률:  -0.2227446814127776 -0.1517
ExtraTreesRegressor 의 정답률:  0.1407902263316413 0.1965
GammaRegressor 의 정답률:  0.036102875217516206 0.023
GaussianProcessRegressor 의 정답률:  -25.884086484688297 -10.3071
GradientBoostingRegressor 의 정답률:  0.32520755913548105 0.3279
HistGradientBoostingRegressor 의 정답률:  0.3442698002031558 0.3495
HuberRegressor 의 정답률:  0.2340639873204865 0.2374
IsotonicRegression 은 에러
KNeighborsRegressor 의 정답률:  0.2637461691063886 0.2999
KernelRidge 의 정답률:  0.21879428218513763 0.2345
Lars 의 정답률:  0.2494896826312223 0.2603
LarsCV 의 정답률:  0.24955713996545115 0.2601
Lasso 의 정답률:  0.24789006679844872 0.258
LassoCV 의 정답률:  0.24954124639126518 0.2603
LassoLars 의 정답률:  -0.0006494197429334214 -0.0011
LassoLarsCV 의 정답률:  0.24955713996545115 0.2601
LassoLarsIC 의 정답률:  0.2495246227984803 0.2604
LinearRegression 의 정답률:  0.2494896826312223 0.2603
LinearSVR 의 정답률:  0.1931759104992573 0.1837
MLPRegressor 의 정답률:  0.2549474815413063 0.2616
MultiOutputRegressor 은 에러
MultiTaskElasticNet 은 에러
MultiTaskElasticNetCV 은 에러
MultiTaskLasso 은 에러
MultiTaskLassoCV 은 에러
NuSVR 의 정답률:  0.20753327079523343 0.1908
OrthogonalMatchingPursuit 의 정답률:  0.13937599050521832 0.1577
OrthogonalMatchingPursuitCV 의 정답률:  0.24823627471987486 0.2585
PLSCanonical 의 정답률:  -0.6780629428885865 -0.5768
PLSRegression 의 정답률:  0.24308584929802923 0.2541
PassiveAggressiveRegressor 의 정답률:  0.2328115210822176 0.2337
PoissonRegressor 의 정답률:  0.2418083239096911 0.2667
"""

