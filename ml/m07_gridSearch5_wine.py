import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_wine()
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
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold, verbose=1, 
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
print("accuracy_score: ", accuracy_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best))  

print("걸린 시간: ", end - start)

''' 
최적의 매개변수:  RandomForestClassifier(max_depth=10)  
최적의 파라미터:  {'max_depth': 10, 'n_estimators': 100}
best_score_ :  0.9788177339901478
model.score:  1.0
accuracy_score:  1.0
최적 튠 ACC:  1.0
걸린 시간:  10.052372217178345
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
0               {'max_depth': 6, 'n_estimators': 100}         0.964532               13           0.965517
1               {'max_depth': 6, 'n_estimators': 200}         0.964532               13           0.965517
2               {'max_depth': 8, 'n_estimators': 100}         0.964532               13           0.965517
3               {'max_depth': 8, 'n_estimators': 200}         0.971675                5           0.965517
4              {'max_depth': 10, 'n_estimators': 100}         0.978818                1           0.965517
5              {'max_depth': 10, 'n_estimators': 200}         0.964532               13           0.965517
6              {'max_depth': 12, 'n_estimators': 100}         0.971675                5           0.965517
7              {'max_depth': 12, 'n_estimators': 200}         0.971675                5           0.965517
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.964532               13           0.965517
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.950246               40           0.965517
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.950246               40           0.965517
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.922660               56           0.896552
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.950246               40           0.965517
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.964532               13           0.965517
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.950493               37           0.965517
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.950493               37           0.965517
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.950246               40           0.965517
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.943350               50           0.965517
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.950246               40           0.965517
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.936453               51           0.965517
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.957882               26           0.965517
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.943596               47           0.896552
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.943596               47           0.965517
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.943842               45           0.931034
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.964532               13           0.965517
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.950739               36           0.965517
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.964286               24           1.000000
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.957389               30           0.965517
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.957143               35           1.000000
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.957389               30           0.965517
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.957635               27           0.931034
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.957389               30           0.965517
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.957389               30           0.965517
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.936453               51           0.896552
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.943596               47           0.965517
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.936453               51           0.965517
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.950493               37           0.965517
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.936453               51           0.931034
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.943842               45           0.862069
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.929310               55           0.931034
40           {'max_depth': 6, 'min_samples_split': 2}         0.971675                5           0.965517
41           {'max_depth': 6, 'min_samples_split': 3}         0.957389               30           0.965517
42           {'max_depth': 6, 'min_samples_split': 5}         0.971675                5           1.000000
43          {'max_depth': 6, 'min_samples_split': 10}         0.964532               13           0.965517
44           {'max_depth': 8, 'min_samples_split': 2}         0.978818                1           0.965517
45           {'max_depth': 8, 'min_samples_split': 3}         0.964286               24           1.000000
46           {'max_depth': 8, 'min_samples_split': 5}         0.964532               13           0.965517
47          {'max_depth': 8, 'min_samples_split': 10}         0.971675                5           0.965517
48          {'max_depth': 10, 'min_samples_split': 2}         0.964778               12           0.965517
49          {'max_depth': 10, 'min_samples_split': 3}         0.957635               27           0.965517
50          {'max_depth': 10, 'min_samples_split': 5}         0.964532               13           0.965517
51         {'max_depth': 10, 'min_samples_split': 10}         0.964532               13           0.965517
52          {'max_depth': 12, 'min_samples_split': 2}         0.957635               27           0.965517
53          {'max_depth': 12, 'min_samples_split': 3}         0.978571                4           1.000000
54          {'max_depth': 12, 'min_samples_split': 5}         0.971675                5           0.965517
55         {'max_depth': 12, 'min_samples_split': 10}         0.978818                1           0.965517
'''




