from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # (다수를 차지하는 것을 정답으로 사용)
from sklearn.linear_model import LogisticRegression  # 분류모델 (0과 1 사이)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1) 데이터

datasets = load_breast_cancer()

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
1. Perceptron:  0.8947368421052632
accuracy:  0.8947368421052632 
2. LinearSVC:  0.8596491228070176
accuracy:  0.8596491228070176
3. SVC:  0.8947368421052632
accuracy:  0.8947368421052632
4.KNeighborsClassifier:  0.9210526315789473
accuracy:  0.9210526315789473
5.LogisticRegression:  0.956140350877193
accuracy:  0.956140350877193
6.DecisionTreeClassifier:  0.9210526315789473
accuracy:  0.9210526315789473
7.RandomForestClassifier:  0.9649122807017544
accuracy:  0.9649122807017544
"""