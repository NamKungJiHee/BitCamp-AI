import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

'''
Stratified K-fold: K fold는 random으로 데이터 셋을 split 해주는데, 이 때문에 레이블 값의 분포(비율)가 기존 데이터 full 셋에서의 분포(비율)와 크게 달라질 수도 있다.

Stratified K-fold 교차 검증 방법은 원본 데이터에서 레이블 분포를 먼저 고려한 뒤, 이 분포와 동일하게 학습 및 검증 데이터 세트를 분배한다.

(예를 들면  A와 B로 이루어진 원본 데이터의 구성 비율이 A : B = 3 : 7 이라면, training set 및 test set의 데이터의 구성비율도 A : B = 3 : 7이 되게 만들어 주는 개념이다.)\
    
#Stratified K 폴드는 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K 폴드 방식으로, 원본 데이터 집합의 레이블 분포를 먼저 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트를 분배
# 일반적으로 분류에서의 교차검증에는 Stratified K 폴드를 사용해야하지만, 회귀에서는 동 방식이 지원되지 않는다. 회귀의 결정값은 연속된 숫자값이기 때문에 결정값별로 분포를 정하는 의미가 없기 때문    
'''

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold # 주로 분류에서 사용함

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
#kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)
kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 66)

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()
#model = Perceptron()
#model = LinearSVC() 
#model = KNeighborsClassifier()
#model = LogisticRegression()
#model = DecisionTreeClassifier()
#model = RandomForestClassifier()


scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4)) # np.mean(scores)  5번 교차검증한 결과의 평균값! # 소수점4까지 잘라주기

#ACC:  [0.87912088 0.97802198 0.92307692 0.89010989 0.9010989 ]
#cross_val_score:  0.9143

"""
1. SVC:ACC:  [0.92307692 0.92307692 0.9010989  0.91208791 0.91208791] 
 cross_val_score:  0.9143  
2. perceptron:ACC:  [0.91208791 0.82417582 0.84615385 0.81318681 0.76923077] 
 cross_val_score:  0.833 
3. LinearSVC: ACC:  [0.91208791 0.93406593 0.93406593 0.92307692 0.92307692]
 cross_val_score:  0.9253
4.KNeighbors: ACC:  [0.94505495 0.92307692 0.94505495 0.94505495 0.91208791] 
 cross_val_score:  0.9341
5.LogisticRegression: ACC:  [0.94505495 0.91208791 0.95604396 0.95604396 0.94505495]
 cross_val_score:  0.9429
6.DecisionTreeClassifier: ACC:  [0.91208791 0.91208791 0.93406593 0.89010989 0.87912088] 
 cross_val_score:  0.9055
7. RandomForestClassifier: ACC:  [0.94505495 0.97802198 0.98901099 0.93406593 0.92307692]
 cross_val_score:  0.9538
"""