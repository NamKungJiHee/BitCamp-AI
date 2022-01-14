from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

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

print("allAlgorithms: ", allAlgorithms)  # [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ..]
print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수: 55

for (name, algorithm) in allAlgorithms:   # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),...
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률: ', r2)
    except:                     # 에러나는 것 빼고 계속해라.
        #continue   
        print(name, '은 에러') 
""" 
모델의 갯수:  55
ARDRegression 의 정답률:  0.24926480107309645
AdaBoostRegressor 의 정답률:  0.17430941250848364
BaggingRegressor 의 정답률:  0.23813241432026078
BayesianRidge 의 정답률:  0.24957561720517973
CCA 의 정답률:  -0.1875185306363374
DecisionTreeRegressor 의 정답률:  -0.19637527382024778
DummyRegressor 의 정답률:  -0.0006494197429334214
ElasticNet 의 정답률:  0.060591067855856995
ElasticNetCV 의 정답률:  0.24125986412862332
ExtraTreeRegressor 의 정답률:  -0.14891489029146698
ExtraTreesRegressor 의 정답률:  0.14717167684307997
GammaRegressor 의 정답률:  0.036102875217516206
GaussianProcessRegressor 의 정답률:  -25.884086484688297
GradientBoostingRegressor 의 정답률:  0.3252091901960348
HistGradientBoostingRegressor 의 정답률:  0.3442698002031558
HuberRegressor 의 정답률:  0.2340639873204865
IsotonicRegression 은 에러
KNeighborsRegressor 의 정답률:  0.2637461691063886
KernelRidge 의 정답률:  0.21879428218513763
Lars 의 정답률:  0.2494896826312223
LarsCV 의 정답률:  0.24955713996545115
Lasso 의 정답률:  0.24789006679844872
LassoCV 의 정답률:  0.24954124639126518
LassoLars 의 정답률:  -0.0006494197429334214
LassoLarsCV 의 정답률:  0.24955713996545115
LassoLarsIC 의 정답률:  0.2495246227984803
LinearRegression 의 정답률:  0.2494896826312223
LinearSVR 의 정답률:  0.19016413659273446
MLPRegressor 의 정답률:  0.258615227461679
MultiOutputRegressor 은 에러
MultiTaskElasticNet 은 에러
MultiTaskElasticNetCV 은 에러
MultiTaskLasso 은 에러
MultiTaskLassoCV 은 에러
NuSVR 의 정답률:  0.20753327079523343
OrthogonalMatchingPursuit 의 정답률:  0.13937599050521832
OrthogonalMatchingPursuitCV 의 정답률:  0.24823627471987486
PLSCanonical 의 정답률:  -0.6780629428885865
PLSRegression 의 정답률:  0.24308584929802923
PassiveAggressiveRegressor 의 정답률:  0.219325711922864
PoissonRegressor 의 정답률:  0.2418083239096911
"""