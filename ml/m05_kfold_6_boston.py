import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

datasets =  load_boston()

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

#model = KNeighborsRegressor()
#model = LogisticRegression()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4)) # np.mean(scores)  5번 교차검증한 결과의 평균값! # 소수점4까지 잘라주기


'''
1. KNeighborsRegressor : ACC:  [0.38689566 0.52994483 0.3434155  0.55325748 0.51995804] 
 cross_val_score:  0.4667
2. LinearRegression : ACC:  [0.5815212  0.69885237 0.6537276  0.77449543 0.70223459] 
 cross_val_score:  0.6822
3. DecisionTreeRegressor : ACC:  [0.69746901 0.62099578 0.62508169 0.70334893 0.75770817] 
 cross_val_score:  0.6809
4. RandomForestRegressor : ACC:  [0.86460397 0.73776355 0.80461213 0.86702561 0.89982663] 
 cross_val_score:  0.8348
'''