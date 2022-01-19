import numpy as np, pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   

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
best_score_ :  0.35734324644467313
model.score:  0.3511802559662027
r2_score:  0.3511802559662027
최적 튠 R2:  0.3511802559662027
걸린 시간:  142.83545637130737
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
0               {'max_depth': 6, 'n_estimators': 100}         0.332447               50           0.314335
1               {'max_depth': 6, 'n_estimators': 200}         0.332443               51           0.312991
2               {'max_depth': 8, 'n_estimators': 100}         0.351245                9           0.333135
3               {'max_depth': 8, 'n_estimators': 200}         0.350233               13           0.330393
4              {'max_depth': 10, 'n_estimators': 100}         0.355681                4           0.328759
5              {'max_depth': 10, 'n_estimators': 200}         0.357309                2           0.334074
6              {'max_depth': 12, 'n_estimators': 100}         0.347553               30           0.311578
7              {'max_depth': 12, 'n_estimators': 200}         0.346790               35           0.310202
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.331590               52           0.311896
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.330517               54           0.312571
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.330063               55           0.310625
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.329261               56           0.310547
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.348265               27           0.330961
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.345779               40           0.327796
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.345544               41           0.328508
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.343308               43           0.323590
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.353679                7           0.330125
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.351241               10           0.327441
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.349694               17           0.329436
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.346407               36           0.326078
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.350265               12           0.316489
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.351068               11           0.323898
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.349400               20           0.323766
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.346263               38           0.327233
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.340146               45           0.299772
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.342421               44           0.304258
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.339060               46           0.299504
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.346365               37           0.315435
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.347357               32           0.318252
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.346887               34           0.318517
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.346192               39           0.318138
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.348339               26           0.322075
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.348695               24           0.324006
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.349190               22           0.324558
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.348511               25           0.324991
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.348259               28           0.323954
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.349225               21           0.330941
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.348894               23           0.330580
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.349521               18           0.329945
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.347239               33           0.328464
40           {'max_depth': 6, 'min_samples_split': 2}         0.332558               48           0.312170
41           {'max_depth': 6, 'min_samples_split': 3}         0.332505               49           0.313797
42           {'max_depth': 6, 'min_samples_split': 5}         0.332797               47           0.314812
43          {'max_depth': 6, 'min_samples_split': 10}         0.331264               53           0.312839
44           {'max_depth': 8, 'min_samples_split': 2}         0.349835               15           0.332489
46           {'max_depth': 8, 'min_samples_split': 5}         0.349746               16           0.332421
47          {'max_depth': 8, 'min_samples_split': 10}         0.348041               29           0.332464
48          {'max_depth': 10, 'min_samples_split': 2}         0.354080                5           0.328713
49          {'max_depth': 10, 'min_samples_split': 3}         0.357343                1           0.334428
50          {'max_depth': 10, 'min_samples_split': 5}         0.356167                3           0.329856
51         {'max_depth': 10, 'min_samples_split': 10}         0.353952                6           0.332011
52          {'max_depth': 12, 'min_samples_split': 2}         0.344206               42           0.306262
53          {'max_depth': 12, 'min_samples_split': 3}         0.347409               31           0.315473
54          {'max_depth': 12, 'min_samples_split': 5}         0.349488               19           0.317404
55         {'max_depth': 12, 'min_samples_split': 10}         0.351322                8           0.323947
'''