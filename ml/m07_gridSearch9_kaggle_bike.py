import numpy as np, pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_split=10)
최적의 파라미터:  {'max_depth': 6, 'min_samples_split': 10}
best_score_ :  0.018718144355323194
model.score:  0.014233241505968778
accuracy_score:  0.014233241505968778
최적 튠 ACC:  0.014233241505968778
걸린 시간:  165.75570631027222
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
0               {'max_depth': 6, 'n_estimators': 100}         0.017684                4           0.018370
1               {'max_depth': 6, 'n_estimators': 200}         0.018029                3           0.018944
2               {'max_depth': 8, 'n_estimators': 100}         0.014814               15           0.013777
3               {'max_depth': 8, 'n_estimators': 200}         0.014125               20           0.011481
4              {'max_depth': 10, 'n_estimators': 100}         0.011024               38           0.010333
5              {'max_depth': 10, 'n_estimators': 200}         0.011828               34           0.009759
6              {'max_depth': 12, 'n_estimators': 100}         0.010221               42           0.009759
7              {'max_depth': 12, 'n_estimators': 200}         0.010221               43           0.008037
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.016306               11           0.020092
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.016306               10           0.017796
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.017455                6           0.020666
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.017225                7           0.016648
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.015044               14           0.013203
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.015273               13           0.016073
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.014240               17           0.010333
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.016307                9           0.013203
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.012402               29           0.010907
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.012747               27           0.010333
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.013895               23           0.013777
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.014010               21           0.010333
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.009761               47           0.006889
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.011483               37           0.010907
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.012287               31           0.009759
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.014125               19           0.013777
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.007924               56           0.008611
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.008842               54           0.006315
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.007924               55           0.007463
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.009416               51           0.009185
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.008842               53           0.005741
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.009991               46           0.008611
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.009646               49           0.007463
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.009417               50           0.008037
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.010450               40           0.009185
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.009991               45           0.006889
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.010335               41           0.008037
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.011598               36           0.006889
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.012632               28           0.008037
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.011943               33           0.008611
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.012862               25           0.011481
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.012976               24           0.009185
40           {'max_depth': 6, 'min_samples_split': 2}         0.016881                8           0.015499
41           {'max_depth': 6, 'min_samples_split': 3}         0.017570                5           0.016648
42           {'max_depth': 6, 'min_samples_split': 5}         0.018258                2           0.020092
43          {'max_depth': 6, 'min_samples_split': 10}         0.018718                1           0.018944
44           {'max_depth': 8, 'min_samples_split': 2}         0.014240               17           0.013777
45           {'max_depth': 8, 'min_samples_split': 3}         0.014010               22           0.013777
46           {'max_depth': 8, 'min_samples_split': 5}         0.014814               16           0.014351
47          {'max_depth': 8, 'min_samples_split': 10}         0.015273               12           0.013777
48          {'max_depth': 10, 'min_samples_split': 2}         0.012173               32           0.012055
49          {'max_depth': 10, 'min_samples_split': 3}         0.011713               35           0.010333
50          {'max_depth': 10, 'min_samples_split': 5}         0.012402               30           0.009759
51         {'max_depth': 10, 'min_samples_split': 10}         0.012747               26           0.010907
52          {'max_depth': 12, 'min_samples_split': 2}         0.009187               52           0.008611
53          {'max_depth': 12, 'min_samples_split': 3}         0.010106               44           0.009185
54          {'max_depth': 12, 'min_samples_split': 5}         0.009646               48           0.009759
55         {'max_depth': 12, 'min_samples_split': 10}         0.010680               39           0.012055
'''