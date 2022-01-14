import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# 분류, 회귀모델

#1) 데이터
datasets = load_iris()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)



#2) 모델구성 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsClassifier()
#model = LogisticRegression()
#model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3) 훈련

model.fit(x_train, y_train) # 훈련시켜줌

#4) 평가, 예측
# loss=model.evaluate(x_test, y_test)  
# print('loss: ', loss[0]) 
# print('accuracy: ', loss[1]) 
result = model.score(x_test, y_test)  # 결과가 accuracy로 나옴 # metrics의 개념(accuracy값을 돌려준다.)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)   #(,) 값 비교

print("RandomForestClassifier: ", result)
print("accuracy: ", acc)

""" 
1. Perceptron:  0.9333333333333333
accuracy:  0.9333333333333333
2. LinearSVC:  0.9666666666666667
accuracy:  0.9666666666666667
3. SVC:  0.9666666666666667
accuracy:  0.9666666666666667
4. KNeighborsClassifier:  0.9666666666666667
accuracy:  0.9666666666666667
5. LogisticRegression:  1.0
accuracy:  1.0
6.DecisionTreeClassifier:  0.9333333333333333
accuracy:  0.9333333333333333
7.RandomForestClassifier:  0.9333333333333333
accuracy:  0.9333333333333333
"""