import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']
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
1. SVC: ACC:  [0.02353617 0.01549943 0.02238806 0.01033889 0.01608271]
 cross_val_score:  0.0176
2. Perceptron : ACC:  [0.01262916 0.00114811 0.00401837 0.00459506 0.01263642] 
 cross_val_score:  0.007
3. KNeighborsClassifier : ACC:  [0.16794866 0.18517249 0.23680429 0.16578537 0.23294712] 
 cross_val_score:  0.1977
4. LinearRegression : ACC:  [0.2369219  0.26392691 0.29226917 0.24560392 0.26296619] 
 cross_val_score:  0.2603
5. DecisionTreeRegressor:  ACC:  [-0.31225364 -0.03070913 -0.11201453 -0.2916193  -0.23565108] 
 cross_val_score:  -0.1964
6. RandomForestClassifier : ACC:  [0.21803057 0.30177941 0.32938996 0.27931993 0.31075132]
 cross_val_score:  0.2879
'''