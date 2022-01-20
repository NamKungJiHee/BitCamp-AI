import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

#x = np.delete(x,[1,2,7],axis=1) 
x = np.delete(x,[0,2,3],axis=1) 

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

print("accuracy: ", acc1)  #accuracy:  .9722222222222222
print("accuracy: ", acc2)  #accuracy:   1.0 
print("accuracy: ", acc3)  #accuracy:  1.0
print("accuracy: ", acc4)  #accuracy:   0.9722222222222222

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  


''' 
[0.00489447 0.         0.         0.0555874  0.01598859 0.
 0.1569445  0.         0.         0.04078249 0.03045446 0.33215293
 0.36319516]
[0.11764062 0.0313699  0.01827036 0.04178866 0.02032958 0.04699689
 0.12261865 0.01157328 0.03685689 0.15358379 0.08800363 0.12109578
 0.18987196]
[0.01854127 0.04139537 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707159 0.01631111 0.00051476 0.12775213 0.01918284 0.50344414
 0.10358089]
[1.72086814e-02 4.27219424e-02 2.03106077e-02 6.85036805e-03
 3.22317610e-03 3.07795893e-05 1.03373259e-01 6.76822746e-04
 9.70233612e-05 2.49446718e-01 2.86746042e-02 2.50311296e-01
 2.77074721e-01]
 
 기존 acc
1. Decision Tree
("accuracy: ", acc1)  #accuracy:  .9722222222222222
컬럼 삭제 후 acc
accuracy:   0.9444444444444444

2. RandomForestClassifier
("accuracy: ", acc2)  #accuracy:   1.0 
accuracy:    1.0

3. XGBClassifier
("accuracy: ", acc3)  #accuracy:  1.0 
accuracy:  0.9722222222222222

4. GradientBoostingClassifier
("accuracy: ", acc4)  #accuracy:  0.9722222222222222
accuracy:  0.9722222222222222
'''