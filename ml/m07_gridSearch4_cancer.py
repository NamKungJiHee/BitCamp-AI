import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10]},
    {'min_samples_leaf' : [3, 5, 7,10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]}]

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
Fitting 5 folds for each of 44 candidates, totalling 220 fits
최적의 매개변수:  RandomForestClassifier(max_depth=8, n_estimators=200)
최적의 파라미터:  {'max_depth': 8, 'n_estimators': 200}
best_score_ :  0.9648351648351647
model.score:  0.956140350877193
accuracy_score:  0.956140350877193
최적 튠 ACC:  0.956140350877193
걸린 시간:  8.424423933029175
'''
###################################################################

#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)
''' 
                                  params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.958242               10           0.923077
1               {'max_depth': 6, 'n_estimators': 200}         0.956044               19           0.923077
2               {'max_depth': 8, 'n_estimators': 100}         0.964835                1           0.945055
3               {'max_depth': 8, 'n_estimators': 200}         0.962637                2           0.934066
4              {'max_depth': 10, 'n_estimators': 100}         0.960440                6           0.934066
5              {'max_depth': 10, 'n_estimators': 200}         0.960440                6           0.934066
6              {'max_depth': 12, 'n_estimators': 100}         0.962637                2           0.934066
7              {'max_depth': 12, 'n_estimators': 200}         0.962637                2           0.923077
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.958242               10           0.923077
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.958242               10           0.934066
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.951648               31           0.912088
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.949451               38           0.901099
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.958242               10           0.934066
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.951648               31           0.934066
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.956044               19           0.912088
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.951648               31           0.923077
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.953846               26           0.934066
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.953846               26           0.934066
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.956044               19           0.923077
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.951648               31           0.901099
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.958242               10           0.934066
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.956044               19           0.934066
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.953846               26           0.901099
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.949451               38           0.912088
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.958242               10           0.923077
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.953846               26           0.923077
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.956044               19           0.923077
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.956044               19           0.923077
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.958242               10           0.923077
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.958242               10           0.923077
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.953846               26           0.912088
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.960440                6           0.923077
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.949451               38           0.923077
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.951648               31           0.901099
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.945055               42           0.901099
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.951648               31           0.901099
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.945055               42           0.912088
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.951648               31           0.901099
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.945055               42           0.901099
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.949451               38           0.912088
40                           {'min_samples_split': 2}         0.962637                2           0.945055
41                           {'min_samples_split': 3}         0.960440                6           0.923077
42                           {'min_samples_split': 5}         0.958242               10           0.923077
43                          {'min_samples_split': 10}         0.956044               19           0.923077
'''




