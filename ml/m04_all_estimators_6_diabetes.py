from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

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
ARDRegression 의 정답률:  0.8119016106669674
AdaBoostRegressor 의 정답률:  0.8914048635687275
BaggingRegressor 의 정답률:  0.9071394697276283
BayesianRidge 의 정답률:  0.8119880571377842
CCA 의 정답률:  0.791347718442463
DecisionTreeRegressor 의 정답률:  0.6800794179482337
DummyRegressor 의 정답률:  -0.0005370164400797517
ElasticNet 의 정답률:  0.16201563080833714
ElasticNetCV 의 정답률:  0.8113737663385278
ExtraTreeRegressor 의 정답률:  0.797487750876015
ExtraTreesRegressor 의 정답률:  0.9369144225954446
GammaRegressor 의 정답률:  0.1964792057029865
GaussianProcessRegressor 의 정답률:  -1.578958693000398
GradientBoostingRegressor 의 정답률:  0.9459159972410115
HistGradientBoostingRegressor 의 정답률:  0.9323326124661162
HuberRegressor 의 정답률:  0.7958373063951082
IsotonicRegression 은 에러
KNeighborsRegressor 의 정답률:  0.8265307833211177
KernelRidge 의 정답률:  0.803254958502079
Lars 의 정답률:  0.7746736096721598
LarsCV 의 정답률:  0.7981576314184016
Lasso 의 정답률:  0.242592140544296
LassoCV 의 정답률:  0.8125908596954046
LassoLars 의 정답률:  -0.0005370164400797517
LassoLarsCV 의 정답률:  0.8127604328474283
LassoLarsIC 의 정답률:  0.8131423868817642
LinearRegression 의 정답률:  0.8111288663608667
LinearSVR 의 정답률:  0.709246273626764
MLPRegressor 의 정답률:  0.4483377137312884
MultiOutputRegressor 은 에러
MultiTaskElasticNet 은 에러
MultiTaskElasticNetCV 은 에러
MultiTaskLasso 은 에러
MultiTaskLassoCV 은 에러
NuSVR 의 정답률:  0.6254681434531
OrthogonalMatchingPursuit 의 정답률:  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률:  0.78617447738729
PLSCanonical 의 정답률:  -2.2317079741425734
PLSRegression 의 정답률:  0.8027313142007888
PassiveAggressiveRegressor 의 정답률:  0.7123655573274266
PoissonRegressor 의 정답률:  0.67496007101481
QuantileRegressor 의 정답률:  -0.020280478327147522
RANSACRegressor 의 정답률:  0.019454324679768153        
RadiusNeighborsRegressor 의 정답률:  0.41191760158788593
RandomForestRegressor 의 정답률:  0.9248871891421689
RegressorChain 은 에러
Ridge 의 정답률:  0.8087497007195745
RidgeCV 의 정답률:  0.8116598578372426
SGDRegressor 의 정답률:  0.823389689389382
SVR 의 정답률:  0.6597910766772523
StackingRegressor 은 에러
TheilSenRegressor 의 정답률:  0.7838462105177335
TransformedTargetRegressor 의 정답률:  0.8111288663608667
TweedieRegressor 의 정답률:  0.19473445117356525
VotingRegressor 은 에러
"""