import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x = np.delete(x,[0,1,2,3,4,5,6,7],axis=1) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier(max_depth=5)
model2= RandomForestClassifier(max_depth=5)
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

#3) 훈련

model1.fit(x_train, y_train) 
model2.fit(x_train, y_train) 
model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 

#4) 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
acc1 = accuracy_score(y_test, y_predict1) 
y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test, y_predict2) 
y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test, y_predict3) 
y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test, y_predict4) 

print("accuracy: ", acc1)  #accuracy:  0.9210526315789473
print("accuracy: ", acc2)  #accuracy:   0.956140350877193 
print("accuracy: ", acc3)  #accuracy:  0.9736842105263158
print("accuracy: ", acc4)  #accuracy:   0.956140350877193

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  

''' 
기존 acc
1. Decision Tree
("accuracy: ", acc1)  #accuracy:  0.9210526315789473
컬럼 삭제 후 acc
accuracy:  0.9210526315789473

2. RandomForestClassifier
("accuracy: ", acc2)  #accuracy:  0.956140350877193 
accuracy:   0.956140350877193

3. XGBClassifier
("accuracy: ", acc3)  #accuracy:  0.9736842105263158
accuracy:  0.9736842105263158
4. GradientBoostingClassifier
("accuracy: ", acc4)  #accuracy:  0.956140350877193
accuracy:  0.9298245614035088
'''
