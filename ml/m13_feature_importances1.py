import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
''' 
'XGBoost (Extreme Gradient Boosting)' 는 앙상블의 부스팅 기법의 한 종류

이전 모델의 오류를 순차적으로 보완해나가는 방식
'''
#1) 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#model = DecisionTreeClassifier(max_depth=5)
#model = RandomForestClassifier(max_depth=5)
#model = XGBClassifier()
model = GradientBoostingClassifier()

#3) 훈련

model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 

#print("RandomForestClassifier: ", result)
print("accuracy: ", acc)

print(model.feature_importances_)

''' 
accuracy:  0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]  # iris는 feature가 4개 / 즉 3번째 feature가 가장 중요!
=====================================================
1. DecisionTreeClassifier
accuracy:  0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]

2. RandomForestClassifier

accuracy:  0.9666666666666667
[0.10056881 0.02463652 0.463239   0.41155567]

3. XGBClassifier

accuracy:  0.9
[0.01835513 0.0256969  0.6204526  0.33549538]

4. GradientBoostingClassifier

accuracy:  0.9333333333333333
[0.00587709 0.01206749 0.27820174 0.70385369]
'''