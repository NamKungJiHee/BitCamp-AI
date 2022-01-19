from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
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
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률: ', r2, round(np.mean(scores),4))
    except:                     
        #continue   
        print(name, '은 에러')     
    
""" 
ARDRegression 의 정답률:  0.498748289056254 0.4906
AdaBoostRegressor 의 정답률:  0.36346103354314485 0.4795
BaggingRegressor 의 정답률:  0.35567602131279785 0.4193
BayesianRidge 의 정답률:  0.5014366863847451 0.4869
CCA 의 정답률:  0.48696409064967594 0.4213
DecisionTreeRegressor 의 정답률:  -0.15728231676205162 -0.0307
DummyRegressor 의 정답률:  -0.00015425885559339214 -0.0051
ElasticNet 의 정답률:  0.11987522766332959 0.1261
ElasticNetCV 의 정답률:  0.48941369735908524 0.4826
ExtraTreeRegressor 의 정답률:  -0.11783899309849266 -0.1906
ExtraTreesRegressor 의 정답률:  0.39052026746040125 0.4972
GammaRegressor 의 정답률:  0.07219655012236648 0.0688
GaussianProcessRegressor 의 정답률:  -7.547010959777328 -6.5316
GradientBoostingRegressor 의 정답률:  0.38886067975492267 0.4482
HistGradientBoostingRegressor 의 정답률:  0.28899497703380905 0.4346
HuberRegressor 의 정답률:  0.5068530513878713 0.4829
IsotonicRegression 은 에러
KNeighborsRegressor 의 정답률:  0.3741821819765594 0.4042
KernelRidge 의 정답률:  0.48022687224693617 0.4868
Lars 의 정답률:  0.4919866521464151 0.0658
LarsCV 의 정답률:  0.5010892359535754 0.4585
Lasso 의 정답률:  0.46430753276688697 0.473
LassoCV 의 정답률:  0.4992382182931273 0.4883
LassoLars 의 정답률:  0.3654388741895792 0.3999
LassoLarsCV 의 정답률:  0.4951942790678243 0.4829
LassoLarsIC 의 정답률:  0.49940515175310685 0.4834
LinearRegression 의 정답률:  0.5063891053505036 0.4838
LinearSVR 의 정답률:  0.14990642613604988 0.188
MLPRegressor 의 정답률:  -0.5989194682776209 -0.5355
MultiOutputRegressor 은 에러
MultiTaskElasticNet 은 에러
MultiTaskElasticNetCV 은 에러
MultiTaskLasso 은 에러
MultiTaskLassoCV 은 에러
NuSVR 의 정답률:  0.12527149380257419 0.118
OrthogonalMatchingPursuit 의 정답률:  0.3293449115305741 0.3127
OrthogonalMatchingPursuitCV 의 정답률:  0.44354253337919725 0.4755
PLSCanonical 의 정답률:  -0.9750792277922931 -1.2659
PLSRegression 의 정답률:  0.4766139460349792 0.4836
PassiveAggressiveRegressor 의 정답률:  0.48360028086354234 0.4712
PoissonRegressor 의 정답률:  0.4823231874912104 0.4722
QuantileRegressor 의 정답률:  -0.021939242070499576 -0.0254
RANSACRegressor 의 정답률:  0.39346153565529074 -0.013
RadiusNeighborsRegressor 의 정답률:  0.14407236562185122 0.1338
RandomForestRegressor 의 정답률:  0.3760056286685959 0.4847
RegressorChain 은 에러
Ridge 의 정답률:  0.49950383964954104 0.4869
RidgeCV 의 정답률:  0.49950383964954104 0.4851
SGDRegressor 의 정답률:  0.493224081736348 0.4858
SVR 의 정답률:  0.12343791188320263 0.1196
StackingRegressor 은 에러
TheilSenRegressor 의 정답률:  0.5007337297369095 0.4812
TransformedTargetRegressor 의 정답률:  0.5063891053505036 0.4838
TweedieRegressor 의 정답률:  0.07335459385974419 0.0765
VotingRegressor 은 에러
"""

