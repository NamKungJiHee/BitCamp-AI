import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets =  load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsRegressor()
#model = LogisticRegression()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4)) # np.mean(scores)  5번 교차검증한 결과의 평균값! # 소수점4까지 잘라주기

'''
1. SVC: ACC:  [0.         0.         0.         0.01428571 0.        ] 
 cross_val_score:  0.0029
2. Perceptron : ACC:  [0.         0.01408451 0.         0.01428571 0.02857143] 
 cross_val_score:  0.0114
3. LinearSVC : ACC:  [0.         0.         0.         0.01428571 0.        ] 
 cross_val_score:  0.0029
4. KNeighborsRegressor : ACC:  [0.37000683 0.35477108 0.32086338 0.51614896 0.41040527] 
 cross_val_score:  0.3944
5. LogisticRegression : ACC:  [0.         0.         0.01408451 0.01428571 0.        ] 
 cross_val_score:  0.0057
6. DecisionTreeRegressor : ACC:  [-0.2172678   0.09107625 -0.02190579  0.23580531 -0.13508708] 
 cross_val_score:  -0.0095
7. RandomForestRegressor : ACC:  [0.48310783 0.54167427 0.39042046 0.54273008 0.45639721] 
 cross_val_score:  0.4829
8. LinearRegression: ACC:  [0.53550031 0.49362737 0.47105167 0.55090349 0.36810479] 
 cross_val_score:  0.4838
'''