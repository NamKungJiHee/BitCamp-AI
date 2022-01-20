import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x = np.delete(x,[0,2,3],axis=1)

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

print("accuracy: ", r2_1)  #accuracy:   0.8507309980875364
print("accuracy: ", r2_2)  #accuracy:   0.9116126823892674
print("accuracy: ", r2_3)  #accuracy:  0.9221188601856797
print("accuracy: ", r2_4)  #accuracy:   0.9455782164563826

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  

''' 
[0.03848365 0.         0.         0.         0.01372268 0.29092518
 0.         0.05933307 0.         0.00583002 0.         0.01786395
 0.57384145]
[2.97761170e-02 1.65166833e-04 3.94867646e-03 1.25813908e-03
 1.78788023e-02 3.95277493e-01 6.40488103e-03 7.18881223e-02
 2.57203108e-03 8.35722599e-03 1.23500398e-02 6.26458928e-03
 4.43858716e-01]
[0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
 0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
 0.4284835 ]
[2.45709497e-02 1.98934749e-04 4.68190051e-03 1.80771482e-04
 4.21855980e-02 3.58105743e-01 7.06398274e-03 8.20048390e-02
 2.32173015e-03 1.12499473e-02 3.11501505e-02 6.44972086e-03
 4.29835732e-01]
 
1. 
기존
0.8507309980875364
삭제후 
accuracy:  0.8507309980875364
2.
accuracy:   0.9116126823892674
accuracy:  0.9155597261707573
3.  0.9221188601856797
accuracy:  0.9408336665238235
4.  0.9455782164563826
accuracy:  0.9451729400425323
'''