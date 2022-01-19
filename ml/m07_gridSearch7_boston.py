import numpy as np, pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_boston()
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
최적의 매개변수:  RandomForestRegressor(max_depth=10, min_samples_split=3)
최적의 파라미터:  {'max_depth': 10, 'min_samples_split': 3}
best_score_ :  0.83548315197263
model.score:  0.9228912417197536
r2_score:  0.9228912417197536
최적 튠 R2:  0.9228912417197536
걸린 시간:  11.821555852890015
'''
###################################################################

#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
#print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)

'''                  params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.822283               22           0.865728
1               {'max_depth': 6, 'n_estimators': 200}         0.824137               18           0.872141
2               {'max_depth': 8, 'n_estimators': 100}         0.823290               20           0.868922
3               {'max_depth': 8, 'n_estimators': 200}         0.829636               12           0.871561
4              {'max_depth': 10, 'n_estimators': 100}         0.831194                8           0.871887
5              {'max_depth': 10, 'n_estimators': 200}         0.829753               10           0.872411
6              {'max_depth': 12, 'n_estimators': 100}         0.833859                3           0.877476
7              {'max_depth': 12, 'n_estimators': 200}         0.831788                7           0.863921
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.798709               31           0.795910
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.784706               40           0.763892
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.778076               48           0.762661
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.772511               54           0.747000
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.805606               26           0.807127
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.792087               33           0.767951
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.780403               46           0.767355
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.774909               51           0.744896
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.804062               27           0.809297
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.788764               36           0.768872
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.783474               41           0.771914
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.774240               53           0.756587
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.799035               30           0.803413
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.788180               37           0.770411
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.780976               44           0.765740
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.771797               56           0.738741
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.799718               29           0.794993
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.803495               28           0.811987
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.807727               25           0.816411
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.798593               32           0.797020
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.786408               38           0.763545
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.788987               35           0.774352
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.790379               34           0.770527
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.786288               39           0.774486
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.781326               42           0.764708
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.780031               47           0.767239
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.781082               43           0.762832
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.780424               45           0.754333
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.775790               50           0.744299
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.774262               52           0.752781
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.776044               49           0.741561
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.772271               55           0.749429
40           {'max_depth': 6, 'min_samples_split': 2}         0.822220               23           0.872922
41           {'max_depth': 6, 'min_samples_split': 3}         0.821634               24           0.869128
42           {'max_depth': 6, 'min_samples_split': 5}         0.825181               17           0.855400
43          {'max_depth': 6, 'min_samples_split': 10}         0.822788               21           0.867614
44           {'max_depth': 8, 'min_samples_split': 2}         0.831135                9           0.865392
45           {'max_depth': 8, 'min_samples_split': 3}         0.831954                6           0.874015
46           {'max_depth': 8, 'min_samples_split': 5}         0.823623               19           0.869834
47          {'max_depth': 8, 'min_samples_split': 10}         0.829689               11           0.872006
48          {'max_depth': 10, 'min_samples_split': 2}         0.832919                5           0.870672
49          {'max_depth': 10, 'min_samples_split': 3}         0.835483                1           0.880547
50          {'max_depth': 10, 'min_samples_split': 5}         0.826709               14           0.867055
51         {'max_depth': 10, 'min_samples_split': 10}         0.825375               16           0.870047
52          {'max_depth': 12, 'min_samples_split': 2}         0.833330                4           0.872156
53          {'max_depth': 12, 'min_samples_split': 3}         0.834631                2           0.880621
54          {'max_depth': 12, 'min_samples_split': 5}         0.829249               13           0.880047
55         {'max_depth': 12, 'min_samples_split': 10}         0.825785               15           0.871098
'''
