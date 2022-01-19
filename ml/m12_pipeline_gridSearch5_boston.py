import numpy as np 
from sklearn.datasets import load_boston
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1) 데이터
datasets = load_boston()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#### make_pipeline 사용시 ####

# parameters = [
#     {'randomforestregressor__max_depth' : [6, 8, 10], 'randomforestregressor__min_samples_leaf' : [3, 5, 7]},
#     {'randomforestregressor__min_samples_leaf' : [3, 5, 7], 'randomforestregressor__min_samples_split' : [3, 5, 10]}]
# pipe 자리에는 사용할 model에 대한 parameters가 나와야하므로 (이 때 모델명은 소문자로 써줘야 한다. / _이거는 2번 써준다.)


#### Pipe 사용시 ####
parameters = [
    {'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [3, 5, 7]},
    {'rf__min_samples_leaf' : [3, 5, 7], 'rf__min_samples_split' : [3, 5, 10]}]

#2) 모델구성 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA  # 주성분분석 (고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법)
                                       
#pipe = make_pipeline(StandardScaler(), RandomForestRegressor())  
pipe = Pipeline([("ss",StandardScaler()), ("rf", RandomForestRegressor())])  

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

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 

print("걸린 시간: ", end - start)
print("model.score: ", result)
print("accuracy_score: ", r2)

""" 
## make_pipeline 사용시 ##

1. GridSearchCV

Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  11.05533218383789
model.score:  0.9199067556314204
accuracy_score:  0.9199067556314204

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  6.149906873703003
model.score:  0.9219865886532255
accuracy_score:  0.9219865886532255

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  11.046887874603271
model.score:  0.9158078171990679
accuracy_score:  0.9158078171990679
================================================================
## Pipeline 사용시 ##

1. GridSearchCV

Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  11.04578423500061
model.score:  0.9223634113821941
accuracy_score:  0.9223634113821941

2. RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  6.121081113815308
model.score:  0.9203794849819371
accuracy_score:  0.9203794849819371

3. HalvingGridSearchCV

Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  10.818534135818481
model.score:  0.9165239844218832
accuracy_score:  0.9165239844218832
"""