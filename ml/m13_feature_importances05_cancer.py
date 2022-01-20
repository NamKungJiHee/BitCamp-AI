import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

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

import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautures")
    plt.ylim(-1, n_features)
    
plt.subplot(2,2,1)  # 2행 2열로 뽑아내라 첫번째꺼를
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2)
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3)
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4)
plot_feature_importances_dataset(model4)
plt.show()

''' 
[0.         0.06054151 0.004774   0.00636533 0.         0.
 0.         0.02005078 0.         0.02291518 0.01257413 0.
 0.         0.         0.         0.         0.         0.00442037
 0.         0.         0.         0.01642816 0.         0.72839202
 0.00716099 0.         0.         0.11637753 0.         0.        ]
[0.06434977 0.01122998 0.06063625 0.03091535 0.00504538 0.01548001
 0.0276404  0.10040506 0.00318873 0.00399847 0.03345962 0.00464831
 0.01690799 0.04510217 0.00253928 0.00583583 0.0067636  0.00480203
 0.0028557  0.00365319 0.09369927 0.0208086  0.13777069 0.11807449
 0.00832675 0.01794452 0.03237276 0.1088043  0.00723217 0.00550928]
[0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
 0.0054994  0.09745206 0.00340272 0.00369179 0.00769183 0.00281184
 0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
 0.00639412 0.0050556  0.01813928 0.02285904 0.22248559 0.2849308
 0.00233393 0.         0.00903706 0.11586287 0.00278498 0.00775311]
[5.83622802e-04 3.82956622e-02 2.67736517e-04 2.20294253e-03
 2.16649847e-03 2.19325742e-05 1.48688837e-03 1.23236746e-01
 2.06324864e-03 5.34310473e-04 4.06127466e-03 6.01233973e-05
 5.00323305e-04 1.80687264e-02 3.89517096e-04 3.79735917e-03
 5.31921114e-04 1.66540132e-03 1.45521655e-04 8.69502925e-04
 3.31878815e-01 3.98823890e-02 4.34745197e-02 2.61081031e-01
 4.63117514e-03 9.62929376e-05 1.40102177e-02 1.03279796e-01
 3.19604521e-06 7.13308257e-04]
'''