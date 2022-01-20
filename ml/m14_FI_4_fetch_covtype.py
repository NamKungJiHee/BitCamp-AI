import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

x = np.delete(x,[2],axis=1) 

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

print("accuracy: ", acc1)  #accuracy:  0.7028734197912274
print("accuracy: ", acc2)  #accuracy:  0.6886655249864462
print("accuracy: ", acc3)  #accuracy:   0.869392356479609
print("accuracy: ", acc4)  #accuracy:   0.773499823584589

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  

''' 
[0.82214481 0.00178621 0.         0.02599236 0.0031613  0.01463683
 0.         0.02456075 0.         0.00264821 0.00461784 0.00357592
 0.015303   0.         0.         0.02888085 0.         0.032297
 0.         0.         0.         0.         0.         0.0036305
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.01676442 0.         0.
 0.         0.         0.         0.         0.         0.        ]
[2.69287543e-01 5.81448095e-03 1.10335807e-02 6.76385793e-03
 4.55233594e-03 4.29076846e-02 6.58295826e-03 5.26843665e-03
 4.27750913e-03 2.38805122e-02 2.78762579e-02 1.34611745e-02
 1.48206715e-02 1.93123523e-01 4.63738115e-03 2.06039322e-02
 2.50723824e-03 4.09263072e-02 9.82160369e-04 8.68629640e-03
 1.05070724e-06 0.00000000e+00 1.02198544e-04 4.86592052e-02
 2.20682856e-03 5.92535682e-02 8.29836133e-03 3.20567776e-05
 1.22339664e-07 5.31231011e-05 4.24796231e-04 2.92047717e-04
 6.12635754e-05 7.14034196e-06 0.00000000e+00 6.06766473e-02
 2.26232857e-02 2.04685473e-04 3.83558995e-05 1.09902359e-04
 0.00000000e+00 9.46876065e-05 7.60173778e-03 1.93472312e-03
 3.08227492e-04 1.32071492e-03 3.32000809e-04 0.00000000e+00
 5.91586399e-04 0.00000000e+00 2.66495279e-04 3.14494935e-02
 3.24842306e-02 1.25776212e-02]
[0.08789405 0.00749023 0.00467787 0.01391024 0.00717675 0.0134737
 0.00988278 0.01234562 0.00532679 0.01228409 0.05897678 0.02524585
 0.03400296 0.02194732 0.0036819  0.04537919 0.02427015 0.03912758
 0.00495591 0.00572592 0.00168255 0.0079334  0.00965479 0.01421824
 0.01182582 0.04753365 0.00966461 0.00257953 0.0009051  0.00779884
 0.01206601 0.00594357 0.00512652 0.01335775 0.02019449 0.04943318
 0.02908161 0.01685023 0.00888427 0.00683316 0.01511435 0.00262304
 0.02676792 0.01816068 0.02461869 0.04864944 0.01937804 0.0056233
 0.01617088 0.00294164 0.00907579 0.03714141 0.0415679  0.01282378]
[6.47958420e-01 7.01694862e-03 1.26188355e-03 3.83476409e-02
 9.43554285e-03 5.69098842e-02 7.15471782e-03 2.53972004e-02
 2.76190278e-03 4.16092785e-02 2.54665814e-02 4.98964512e-03
 1.32424182e-02 2.22515945e-03 3.01423557e-04 1.29983880e-02
 4.43319333e-03 1.69606834e-02 9.36700224e-04 1.55418805e-03
 0.00000000e+00 0.00000000e+00 6.43917258e-05 2.37365650e-03
 1.18185981e-03 6.25397649e-03 1.40788404e-03 3.41096450e-04
 5.80363416e-06 4.90335609e-04 1.35560962e-03 5.01191346e-05
 3.26666517e-04 1.44955150e-03 1.03001875e-03 1.65777214e-02
 1.15901253e-02 1.18624297e-03 5.81281617e-05 5.38181436e-05
 3.89661356e-03 8.54018587e-03 8.21391884e-04 7.80996812e-04
 1.61048489e-03 3.95975636e-05 8.54865295e-04 2.98320116e-03
 6.16999630e-03 1.18182694e-03]
 
 
1. 
accuracy:  0.7028734197912274
 
 
2.
accuracy:  0.6886655249864462


3.
accuracy:   0.869392356479609



4. 
accuracy:   0.773499823584589
 
 
'''