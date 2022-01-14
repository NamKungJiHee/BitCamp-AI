import numpy as np
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score   # 교차검증

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

model = SVC()

scores = cross_val_score(model, x, y, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다.  # 분류모델이므로 scores는 accuracy값이다.
print("ACC: ", scores, "\n cross_val_score: ", round(np.mean(scores),4))     # np.mean(scores)  5번 교차검증한 결과의 평균값!

# ACC:  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
# cross_val_score:  0.9667

