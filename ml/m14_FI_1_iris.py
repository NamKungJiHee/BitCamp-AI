import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x = np.delete(x,[0,1],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

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

print("accuracy: ", acc1)  #accuracy:  0.9333333333333333
print("accuracy: ", acc2)  #accuracy:  0.9333333333333333
print("accuracy: ", acc3)  #accuracy:  0.9
print("accuracy: ", acc4)  #accuracy:  0.9333333333333333

print(model1.feature_importances_)  #[0.0125026  0.         0.53835801 0.44913938]
print(model2.feature_importances_)  #[0.11061123 0.02540182 0.42337106 0.44061589]
print(model3.feature_importances_)  #[0.01835513 0.0256969  0.6204526  0.33549538]
print(model4.feature_importances_)  #[0.0042112  0.01380768 0.24108319 0.74089793]



''' 
기존 acc
1. Decision Tree
("accuracy: ", acc1)  #accuracy:  0.9333333333333333
컬럼 삭제 후 acc
accuracy:  0.9333333333333333

2. RandomForestClassifier
("accuracy: ", acc2)  #accuracy:  0.9333333333333333
accuracy:  0.9666666666666667

3. XGBClassifier
("accuracy: ", acc3)  #accuracy:  0.9
accuracy:  0.9666666666666667

4. GradientBoostingClassifier
("accuracy: ", acc4)  #accuracy:  0.9333333333333333
accuracy:  0.9666666666666667
'''
