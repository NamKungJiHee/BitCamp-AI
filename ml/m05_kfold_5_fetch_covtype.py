import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets =  fetch_covtype()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#model = SVC()
#model = Perceptron()
#model = LinearSVC() 
#model = KNeighborsClassifier()
#model = DecisionTreeClassifier()
model = RandomForestClassifier()


scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4)) # np.mean(scores)  5번 교차검증한 결과의 평균값! # 소수점4까지 잘라주기


'''
1. SVC: ACC:  [0.5862069  0.65517241 0.5        0.67857143 0.67857143] 
cross_val_score:  0.6197
2. Perceptron : ACC:  [0.51103677 0.57477249 0.31403154 0.4990964  0.41989652] 
 cross_val_score:  0.4638
3. DecisionTreeClassifier: ACC:  [0.93145586 0.93047697 0.93209053 0.93311245 0.9321651 ]
 cross_val_score:  0.9319
'''