import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_wine()

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
#model = LogisticRegression()
#model = DecisionTreeClassifier()
model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4)) # np.mean(scores)  5번 교차검증한 결과의 평균값! # 소수점4까지 잘라주기

'''
1.SVC: ACC:  [0.5862069  0.65517241 0.5        0.67857143 0.67857143] 
cross_val_score:  0.6197
2. Perceptron: ACC:  [0.55172414 0.62068966 0.53571429 0.35714286 0.60714286] 
 cross_val_score:  0.5345
3. LinearSVC: ACC:  [0.82758621 0.86206897 0.78571429 0.85714286 0.78571429]
 cross_val_score:  0.8236
4. KNeighborsClassifier: ACC:  [0.65517241 0.79310345 0.57142857 0.71428571 0.5       ] 
 cross_val_score:  0.6468
5. LogisticRegression : ACC:  [0.89655172 1.         0.82142857 0.89285714 0.96428571] 
 cross_val_score:  0.915
6. DecisionTreeClassifier : ACC:  [0.79310345 0.75862069 0.82142857 0.78571429 0.89285714] 
 cross_val_score:  0.8103
7. RandomForestClassifier : ACC:  [0.96551724 1.         0.92857143 0.92857143 0.96428571] 
 cross_val_score:  0.9574
'''