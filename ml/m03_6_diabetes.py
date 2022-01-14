from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC    # 되는지 안되는지 check
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  # 분류모델 (0과 1 사이)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1) 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

#2) 모델구성 
#model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsRegressor()
#model = LogisticRegression()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3) 훈련
model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test)  # 결과가 accuracy로 나옴 # metrics의 개념(accuracy값을 돌려준다.)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)  

print("RandomForestRegressor: ", result)
print("r2_score: ", r2)

""" 
1.Perceptron:  0.0
r2_score:  -0.23866809631371377
2.LinearSVC:  0.0
r2_score:  -0.28013728821833284
3.SVC:  0.0
r2_score:  -0.2996632762000788
4.KNeighborsRegressor:  0.38706446604824185
r2_score:  0.38706446604824185
5.LogisticRegression:  0.0
r2_score:  -0.3841885990819991
6.LinearRegression:  0.5209563551242161
r2_score:  0.5209563551242161
7.DecisionTreeRegressor:  -0.17314983837076148
r2_score:  -0.17314983837076148
8.RandomForestRegressor:  0.424651811169358
r2_score:  0.424651811169358
"""