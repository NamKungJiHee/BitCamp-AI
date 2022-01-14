from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # 분류모델 (0과 1 사이)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#1) 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2) 모델구성 
#model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsClassifier()
#model = LogisticRegression()
#model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3) 훈련
model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test)  # 결과가 accuracy로 나옴 # metrics의 개념(accuracy값을 돌려준다.)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)  

print("RandomForestClassifier: ", result)
print("accuracy: ", acc)

""" 
1.Perceptron:  0.6388888888888888
accuracy:  0.6388888888888888
2. LinearSVC:  0.6388888888888888
accuracy:  0.6388888888888888
3. SVC:  0.6944444444444444
accuracy:  0.6944444444444444
4.KNeighborsClassifier:  0.6944444444444444
accuracy:  0.6944444444444444
5.LogisticRegression:  0.9722222222222222
accuracy:  0.9722222222222222
6.DecisionTreeClassifier:  0.9722222222222222
accuracy:  0.9722222222222222
7.RandomForestClassifier:  1.0
accuracy:  1.0
"""