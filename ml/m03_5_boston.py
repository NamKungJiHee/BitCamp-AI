from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC    # 되는지 안되는지 check
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  # 분류모델 (0과 1 사이)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
#회귀
#1) 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2) 모델구성 
model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsRegressor()
#model = LogisticRegression()
#model = LinearRegression()
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()

#3) 훈련
model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test)  

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)  

print("Perceptron: ", result)
print("r2_score: ", r2)


""" 
1.KNeighborsRegressor:  0.5900872726222293
r2_score:  0.5900872726222293
2.LinearRegression:  0.8111288663608656
r2_score:  0.8111288663608656
3.DecisionTreeRegressor:  0.780288678128652
r2_score:  0.780288678128652
4. RandomForestRegressor:  0.921018305173772
r2_score:  0.921018305173772
"""

