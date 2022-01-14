from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

datasets = fetch_covtype()

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
1.Perceptron:  0.49991824651687133 
accuracy:  0.49991824651687133   
2. LinearSVC:  0.40384499539598806
accuracy:  0.40384499539598806
3. SVC:  0.6944444444444444    ============
accuracy:  0.6944444444444444
4. KNeighborsClassifier:  0.9688304088534719
accuracy:  0.9688304088534719
5.LogisticRegression:  0.6198549090815211 
accuracy:  0.6198549090815211
DecisionTreeClassifier:  0.9393130986291232 
accuracy:  0.9393130986291232
7.RandomForestClassifier:  0.9552507250243109 
accuracy:  0.9552507250243109
"""