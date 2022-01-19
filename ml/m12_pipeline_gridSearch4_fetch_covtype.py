import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1) 데이터
datasets = fetch_covtype()
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
                                       
#pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier())  
pipe = Pipeline([("ss",StandardScaler()), ("dt", DecisionTreeClassifier())])  

#model = GridSearchCV(pipe, parameters, cv=5, verbose=1) 
#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1) 
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

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
걸린 시간:  388.66098380088806
model.score:  0.9345541853480547
accuracy_score:  0.9345541853480547

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits                                                                                                                                                       -' 'd:\Study\ml\m12_pipeline_gri
걸린 시간:  206.97860884666443
model.score:  0.9344251009010094
accuracy_score:  0.9344251009010094

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  144.05626916885376
model.score:  0.9345627909778577
accuracy_score:  0.9345627909778577
================================================================
## Pipeline 사용시 ##

1. GridSearchCV

Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  400.540744304657
model.score:  0.9345369740884487
accuracy_score:  0.9345369740884487

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  216.13081169128418  
model.score:  0.9301050747398948
accuracy_score:  0.9301050747398948

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  158.95111846923828
model.score:  0.9344767346798275
accuracy_score:  0.9344767346798275
"""