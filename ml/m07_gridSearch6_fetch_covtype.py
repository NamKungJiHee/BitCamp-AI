import numpy as np, pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = fetch_covtype()
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
Fitting 5 folds for each of 56 candidates, totalling 280 fits
최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3)
최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 2}
best_score_ :  0.938471498622601
model.score:  0.9442785470254641
accuracy_score:  0.9442785470254641
최적 튠 ACC:  0.9442785470254641
걸린 시간:  5067.342813491821 
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
0               {'max_depth': 6, 'n_estimators': 100}         0.686392               54           0.682731
1               {'max_depth': 6, 'n_estimators': 200}         0.689520               48           0.687905
2               {'max_depth': 8, 'n_estimators': 100}         0.721991               38           0.725458
3               {'max_depth': 8, 'n_estimators': 200}         0.722555               37           0.722887
4              {'max_depth': 10, 'n_estimators': 100}         0.752404               28           0.750963
5              {'max_depth': 10, 'n_estimators': 200}         0.751055               31           0.750070
6              {'max_depth': 12, 'n_estimators': 100}         0.781648               18           0.783632
7              {'max_depth': 12, 'n_estimators': 200}         0.779963               22           0.779609
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.688943               49           0.692165
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.685591               56           0.685377
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.686379               55           0.687109
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.687943               50           0.690874
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.720061               40           0.721951
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.718889               46           0.716626
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.720008               41           0.722984
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.721073               39           0.724898
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.750498               35           0.751662
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.750797               32           0.753168
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.750127               36           0.753985
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.750700               34           0.752673
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.780256               21           0.781986
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.778500               24           0.784095
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.777603               25           0.780555
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.776323               26           0.773456
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.938471                1           0.939212
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.938284                2           0.939244
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.938224                3           0.939384
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.934446                4           0.935350
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.928371                5           0.928562
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.927674                8           0.927777
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.927841                7           0.928896
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.927910                6           0.928638
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.919238               10           0.920193
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.919393                9           0.921172
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.919042               12           0.920312
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.908253               15           0.909716
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.908300               14           0.910049
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.908055               16           0.909942
40           {'max_depth': 6, 'min_samples_split': 2}         0.687405               53           0.685925
41           {'max_depth': 6, 'min_samples_split': 3}         0.687880               51           0.685388
42           {'max_depth': 6, 'min_samples_split': 5}         0.689685               47           0.694026
43          {'max_depth': 6, 'min_samples_split': 10}         0.687796               52           0.688443
44           {'max_depth': 8, 'min_samples_split': 2}         0.719622               43           0.721951
45           {'max_depth': 8, 'min_samples_split': 3}         0.719351               45           0.717035
46           {'max_depth': 8, 'min_samples_split': 5}         0.719480               44           0.717454
47          {'max_depth': 8, 'min_samples_split': 10}         0.719980               42           0.720520
48          {'max_depth': 10, 'min_samples_split': 2}         0.753187               27           0.755190
49          {'max_depth': 10, 'min_samples_split': 3}         0.750794               33           0.752598
50          {'max_depth': 10, 'min_samples_split': 5}         0.751177               30           0.751716
51         {'max_depth': 10, 'min_samples_split': 10}         0.751296               29           0.755212
52          {'max_depth': 12, 'min_samples_split': 2}         0.779275               23           0.777909
53          {'max_depth': 12, 'min_samples_split': 3}         0.782111               17           0.784450
54          {'max_depth': 12, 'min_samples_split': 5}         0.780549               19           0.780405
55         {'max_depth': 12, 'min_samples_split': 10}         0.780295               20           0.783051
'''




