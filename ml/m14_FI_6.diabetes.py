import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x = np.delete(x,[1,4],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model1 = DecisionTreeRegressor(max_depth=5)
model2= RandomForestRegressor(max_depth=5)
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()

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

from sklearn.metrics import accuracy_score, r2_score
y_predict1 = model1.predict(x_test)
r2_1 = r2_score(y_test, y_predict1) 
y_predict2 = model2.predict(x_test)
r2_2 = r2_score(y_test, y_predict2) 
y_predict3 = model3.predict(x_test)
r2_3 = r2_score(y_test, y_predict3) 
y_predict4 = model4.predict(x_test)
r2_4 = r2_score(y_test, y_predict4) 

print("accuracy: ", r2_1)  #accuracy: 0.18699053453135217
print("accuracy: ", r2_2)  #accuracy:  0.39792768213539986
print("accuracy: ", r2_3)  #accuracy:  0.23802704693460175
print("accuracy: ", r2_4)  #accuracy:  0.3895615881028167

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  

''' 
기존 / 컬럼 삭제 후 acc
1. DecisionTreeRegressor
accuracy: 0.18699053453135217
accuracy:  0.18699053453135217
2. RandomForestRegressor
accuracy:  0.39792768213539986
accuracy:  0.4048486135865471
3. XGBRegressor
accuracy:  0.23802704693460175
accuracy:  0.22823616776022437
4. GradientBoostingRegressor
accuracy:  0.3895615881028167
accuracy:  0.348023068096308
'''