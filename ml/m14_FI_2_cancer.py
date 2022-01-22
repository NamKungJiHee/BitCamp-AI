#1) 데이터
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x = np.delete(x,[0,1,2,3,4,5,6,7],axis=1) 
# x = pd.DataFrame(x)
# print(type(x))

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
컬럼 삭제 후 acc
1. Decision Tree
accuracy:  0.9210526315789473
accuracy:  0.9385964912280702

2. RandomForestClassifier
accuracy:  0.956140350877193 
accuracy:  0.956140350877193

3. XGBClassifier
accuracy:  0.9736842105263158
accuracy:  0.9736842105263158

4. GradientBoostingClassifier
accuracy:  0.956140350877193
accuracy:  0.9298245614035088
'''
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype

datasets = {'Iris':load_iris(),
            'Wine':load_wine(),
            'Diabets':load_diabetes(),
            'Cancer':load_breast_cancer(),
            'Boston':load_boston(),
            'FetchCov':fetch_covtype(),
            'Kaggle_Bike':'Kaggle_Bike'
            }

model_1 = DecisionTreeClassifier(random_state=66, max_depth=5)
model_1r = DecisionTreeRegressor(random_state=66, max_depth=5)

model_2 = RandomForestClassifier(random_state=66, max_depth=5)
model_2r = RandomForestRegressor(random_state=66, max_depth=5)

model_3 = XGBClassifier(random_state=66)
model_3r = XGBRegressor(random_state=66)

model_4 = GradientBoostingClassifier(random_state=66)
model_4r = GradientBoostingRegressor(random_state=66)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

path = "../_data/kaggle/bike/"  
train = pd.read_csv(path + "train.csv")

model_list = [model_1,model_2,model_3,model_4]
model_list_r = [model_1r,model_2r,model_3r,model_4r]

model_name = ['DecisionTree','RandomForest','XGB','GradientBoosting']

for (dataset_name, dataset) in datasets.items():
    print(f'------------{dataset_name}-----------')
    print('====================================')    
    
    if dataset_name == 'Kaggle_Bike':
        y = train['count']
        x = train.drop(['casual', 'registered', 'count'], axis=1)        
        x['datetime'] = pd.to_datetime(x['datetime'])
        x['year'] = x['datetime'].dt.year
        x['month'] = x['datetime'].dt.month
        x['day'] = x['datetime'].dt.day
        x['hour'] = x['datetime'].dt.hour
        x = x.drop('datetime', axis=1)
        y = np.log1p(y)        
    else:
        x = dataset.data
        y = dataset.target    

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.8, shuffle=True, random_state=66)
    
    plt.figure(figsize=(15,10))
    for i in range(4):        
        plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
        if dataset_name == 'Cancer':
            model_list_r[i].fit(x_train, y_train)
            score = model_list_r[i].score(x_test, y_test)
            feature_importances_ = model_list_r[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list_r[i])    
            
        else: 
            model_list[i].fit(x_train, y_train)
            score = model_list[i].score(x_test, y_test)
            feature_importances_ = model_list[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list[i])    
            plt.ylabel(model_name[i])
            plt.title(dataset_name)

    plt.tight_layout()
    plt.show()
'''