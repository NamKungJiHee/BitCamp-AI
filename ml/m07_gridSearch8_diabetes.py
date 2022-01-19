import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'max_depth' : [6,8,10,12]}]

#2) 모델구성
# model = SVC(C=1, kernel = 'linear', degree=3)
model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose=1, 
                     refit=True, n_jobs=-1)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                                                # Fitting 5 folds for each of 42 candidates, totalling 210 fits
                                                                # refit = True : 가장 좋은 값을 뽑아내겠다.            
                                                                
#3) 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4) 평가, 예측
print("최적의 매개변수: ", model.best_estimator_) 
print("최적의 파라미터: ", model.best_params_) 

print("best_score_ : ", model.best_score_)  
print("model.score: ", model.score(x_test, y_test)) 
 
y_predict = model.predict(x_test)
print("r2_score: ", r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 R2: ", r2_score(y_test, y_pred_best))  

print("걸린 시간: ", end - start)

''' 
최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=5)
최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 5}
best_score_ :  0.508476670694589
model.score:  0.373959632793181
r2_score:  0.373959632793181
최적 튠 R2:  0.373959632793181
걸린 시간:  37.722819089889526
'''
###################################################################

#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
#print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)

''' 
               params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.478975               53           0.487169
1               {'max_depth': 6, 'n_estimators': 200}         0.492998               11           0.499747
2               {'max_depth': 8, 'n_estimators': 100}         0.489886               19           0.505443
3               {'max_depth': 8, 'n_estimators': 200}         0.485379               33           0.477229
4              {'max_depth': 10, 'n_estimators': 100}         0.485687               32           0.488778
5              {'max_depth': 10, 'n_estimators': 200}         0.483262               43           0.479227
6              {'max_depth': 12, 'n_estimators': 100}         0.484539               38           0.470098
7              {'max_depth': 12, 'n_estimators': 200}         0.484879               35           0.479761
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.486792               29           0.501678
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.494300                7           0.529859
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.486183               31           0.514670
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.485174               34           0.508786
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.484701               37           0.509955
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.489671               20           0.519935
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.498984                2           0.538823
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.480418               51           0.504147
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.486845               27           0.511703
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.496593                5           0.506247
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.491073               13           0.518068
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.493128               10           0.525081
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.493479                8           0.510206
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.490623               17           0.527452
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.490912               16           0.514497
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.478980               52           0.514771
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.478315               54           0.483477
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.490069               18           0.514048
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.508477                1           0.523076
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.498957                3           0.517692
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.495924                6           0.521046
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.484306               39           0.507588
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.481797               48           0.506848
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.491216               12           0.525725
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.490980               15           0.519344
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.487945               26           0.506775
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.489622               21           0.518340
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.484135               40           0.512733
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.488248               24           0.522369
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.486272               30           0.518538
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.482244               46           0.523761
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.481798               47           0.517729
40           {'max_depth': 6, 'min_samples_split': 2}         0.489163               22           0.491212
41           {'max_depth': 6, 'min_samples_split': 3}         0.488807               23           0.491461
42           {'max_depth': 6, 'min_samples_split': 5}         0.491001               14           0.507500
43          {'max_depth': 6, 'min_samples_split': 10}         0.496859                4           0.509406
44           {'max_depth': 8, 'min_samples_split': 2}         0.475434               56           0.460174
45           {'max_depth': 8, 'min_samples_split': 3}         0.484810               36           0.489434
46           {'max_depth': 8, 'min_samples_split': 5}         0.482580               45           0.480528
47          {'max_depth': 8, 'min_samples_split': 10}         0.481567               49           0.483952
48          {'max_depth': 10, 'min_samples_split': 2}         0.480855               50           0.454871
49          {'max_depth': 10, 'min_samples_split': 3}         0.475892               55           0.458777
50          {'max_depth': 10, 'min_samples_split': 5}         0.493471                9           0.482631
51         {'max_depth': 10, 'min_samples_split': 10}         0.482690               44           0.489286
52          {'max_depth': 12, 'min_samples_split': 2}         0.488212               25           0.476213
53          {'max_depth': 12, 'min_samples_split': 3}         0.484116               42           0.475038
54          {'max_depth': 12, 'min_samples_split': 5}         0.484129               41           0.478912
55         {'max_depth': 12, 'min_samples_split': 10}         0.486802               28           0.494289
'''




