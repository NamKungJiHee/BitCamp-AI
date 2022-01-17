import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
''' 
* GridSearchCV 란?
교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에 적용할 수 있게 해주는 사이킷런의 기능 중 하나
====================================================
max_depth 5,6,7  (트리의 깊이)
learning_rate 0.1 , 0.001, 0.0001  
(= optimizer, 최적의 loss값을 구하려고 )
( 너무 크면 튕겨나가고 너무 작으면 소실의 문제)
n_estimate 100,500,1000
'''
#1) 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)


parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},    # 12
    {"C":[1,10,100], "kernel":["rbf"],"gamma":[0.001,0.0001]},       # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                      # 24
    "gamma":[0.01,0.001,0.0001], "degree":[3,4]}]          # 총 42번


#2) 모델구성
# #model = SVC(C=1, kernel = 'linear', degree=3)
model = GridSearchCV(SVC(), parameters, cv = kfold, verbose=1, 
                     refit=True, n_jobs=-1)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                                                # Fitting 5 folds for each of 42 candidates, totalling 210 fits
                                                                # refit = True : 가장 좋은 값을 뽑아내겠다.
                                                                # n_jobs = 다중 cpu  (default = 1) = 많이 쓸수록 속도가 향상됨
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
print("accuracy_score: ", accuracy_score(y_test, y_predict))  # accuracy_score:  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best))  # 최적 튠 ACC:  0.9666666666666667

print("걸린 시간: ", end - start)
'''
best_score_ :  0.9916666666666668   ==> 훈련시킨 train에서의 최적값 (kfold된)
model.score:  0.9666666666666667    ==> test값
accuracy_score:  0.966666666666666  ==> test값
'''

###################################################################
''' 
#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)


# rank 1은 가장 좋다는것 / {'C': 1, 'degree': 3, 'kernel': 'linear'}가 0순위 = 이것이 가장 best다! 

                                               params  mean_test_score  rank_test_score  split0_test_score
0           {'C': 1, 'degree': 3, 'kernel': 'linear'}         0.991667                1           1.000000
1           {'C': 1, 'degree': 4, 'kernel': 'linear'}         0.991667                1           1.000000
2           {'C': 1, 'degree': 5, 'kernel': 'linear'}         0.991667                1           1.000000
3          {'C': 10, 'degree': 3, 'kernel': 'linear'}         0.950000               14           0.916667
4          {'C': 10, 'degree': 4, 'kernel': 'linear'}         0.950000               14           0.916667
5          {'C': 10, 'degree': 5, 'kernel': 'linear'}         0.950000               14           0.916667
6         {'C': 100, 'degree': 3, 'kernel': 'linear'}         0.950000               14           0.916667
7         {'C': 100, 'degree': 4, 'kernel': 'linear'}         0.950000               14           0.916667
8         {'C': 100, 'degree': 5, 'kernel': 'linear'}         0.950000               14           0.916667
.........
'''




