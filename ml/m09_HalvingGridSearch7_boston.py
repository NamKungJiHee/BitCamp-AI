import numpy as np, pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
#1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV 

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'max_depth' : [6,8,10,12]}]  

#2) 모델구성
# #model = SVC(C=1, kernel = 'linear', degree=3)
#model = GridSearchCV(SVC(), parameters, cv = kfold, verbose=1, 
 #                   refit=True, n_jobs=-1, random_state = 66)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                                                # Fitting 5 folds for each of 42 candidates, totalling 210 fits
                                                                # refit = True : 가장 좋은 값을 뽑아내겠다.
                                                                # n_jobs = 다중 cpu  (default = 1) = 많이 쓸수록 속도가 향상됨
                                                                
#model = RandomizedSearchCV(SVC(), parameters, cv = kfold, verbose=1, refit=True, n_jobs=-1, random_state= 66, n_iter = 20)  #20 * 5 (default = 10) / n_iter: 데이터 훈련 횟수                                              
                                                                                                                            # Fitting 5 folds for each of 20 candidates, totalling 100 fits
model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose=1, refit=True, n_jobs=-1)   
                                                                                                                    
#3) 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4) 평가, 예측

# x_text = x_train # 과적합 상황 보여주기
# y_test = y_train # train데이터로 best_estimator_로 예측 뒤 점수를 내면
                   # best_score_나온다.

print("최적의 매개변수: ", model.best_estimator_) # 최적의 매개변수:  SVC(C=1, kernel='linear')
print("최적의 파라미터: ", model.best_params_) # 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_)  # best_score_ :  0.9916666666666668
print("model.score: ", model.score(x_test, y_test)) # model.score:  0.9666666666666667  = model.evaluate과 같다
 
y_predict = model.predict(x_test)
print("accuracy_score: ", r2_score(y_test, y_predict))  # accuracy_score:  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC: ", r2_score(y_test, y_pred_best))  # 최적 튠 ACC:  0.9666666666666667

print("걸린 시간: ", end - start)
 
'''  
 Fitting 5 folds for each of 56 candidates, totalling 280 fits
최적의 매개변수:  RandomForestRegressor(max_depth=12)   
최적의 파라미터:  {'max_depth': 12, 'n_estimators': 100}
best_score_ :  0.8346331367877979
model.score:  0.9130736589577009
accuracy_score:  0.9130736589577009
최적 튠 ACC:  0.9130736589577009
걸린 시간:  7.08167290687561 
=========================================
RandomizedSearchCV (GridSearchCV에 비해 속도가 빠르다 / 성능은 유사) : Fitting 5 folds for each of 10 candidates, totalling 50 fits : 10번만 돈다(100번 이상중에 10개만 뽑아서 돈다.)
Fitting 5 folds for each of 20 candidates, totalling 100 fits
최적의 매개변수:  RandomForestRegressor(max_depth=10, n_estimators=200)
최적의 파라미터:  {'n_estimators': 200, 'max_depth': 10}
best_score_ :  0.8357875480031183
model.score:  0.9239909668084697
accuracy_score:  0.9239909668084697
최적 튠 ACC:  0.9239909668084697   
걸린 시간:  3.712477922439575
===========================================
HalvingGridSearchCV
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 14
max_resources_: 404
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 14
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 42
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_resources: 126
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 378
Fitting 5 folds for each of 3 candidates, totalling 15 fits
최적의 매개변수:  RandomForestRegressor(max_depth=12, n_estimators=200)
최적의 파라미터:  {'max_depth': 12, 'n_estimators': 200}
best_score_ :  0.8168511672787726
model.score:  0.9222713364082966
accuracy_score:  0.9222713364082966
최적 튠 ACC:  0.9222713364082966
걸린 시간:  21.350883960723877
'''