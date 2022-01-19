import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1) 데이터
datasets = load_breast_cancer()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#### make_pipeline 사용시 ####

# parameters = [
#     {'decisiontreeclassifier__max_depth' : [6, 8, 10], 'decisiontreeclassifier__min_samples_leaf' : [3, 5, 7]},
#     {'decisiontreeclassifier__min_samples_leaf' : [3, 5, 7], 'decisiontreeclassifier__min_samples_split' : [3, 5, 10]}]
# pipe 자리에는 사용할 model에 대한 parameters가 나와야하므로 (이 때 모델명은 소문자로 써줘야 한다. / _이거는 2번 써준다.)


#### Pipe 사용시 ####
parameters = [
    {'dt__max_depth' : [6, 8, 10], 'dt__min_samples_leaf' : [3, 5, 7]},
    {'dt__min_samples_leaf' : [3, 5, 7], 'dt__min_samples_split' : [3, 5, 10]}]

#2) 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA  # 주성분분석 (고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법)
                                       
#pipe = make_pipeline(MinMaxScaler(), DecisionTreeClassifier())  
pipe = Pipeline([("mm",MinMaxScaler()), ("dt", DecisionTreeClassifier())])  

model = GridSearchCV(pipe, parameters, cv=5, verbose=1) 
#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1) 
#model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#3) 훈련
import time
start = time.time()
model.fit(x_train, y_train) 
end = time.time()

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict) 

print("걸린 시간: ", end - start)
print("model.score: ", result)
print("accuracy_score: ", acc)

""" 
## make_pipeline 사용시 ##

1. GridSearchCV

Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  0.3375532627105713
model.score:  0.9298245614035088
accuracy_score:  0.9298245614035088

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  0.23735928535461426
model.score:  0.9298245614035088
accuracy_score:  0.9298245614035088

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  0.26529812812805176
model.score:  0.9473684210526315
accuracy_score:  0.9473684210526315
================================================================
## Pipeline 사용시 ##

1. GridSearchCV

Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  0.3472311496734619
model.score:  0.9298245614035088
accuracy_score:  0.9298245614035088

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  0.1795499324798584
model.score:  0.9122807017543859
accuracy_score:  0.9122807017543859

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  0.2674121856689453
model.score:  0.9122807017543859
accuracy_score:  0.9122807017543859
"""