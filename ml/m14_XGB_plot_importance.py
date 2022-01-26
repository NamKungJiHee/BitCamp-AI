from tokenize import PlainToken
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
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
model = XGBClassifier()
#model = GradientBoostingClassifier()

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

import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance

plot_importance(model)
plt.show()

''' 
accuracy:  0.9
[0.01835513 0.0256969  0.6204526  0.33549538]
'''