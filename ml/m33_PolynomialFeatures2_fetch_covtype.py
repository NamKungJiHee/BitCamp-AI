from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape) # (581012, 54)

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 66)

model = make_pipeline(StandardScaler(), RandomForestClassifier())

model.fit(x_train, y_train)

print(round(model.score(x_test, y_test),4)) 
# 0.9572   pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv = 7, scoring = 'r2')
print(scores)
# [0.8851758  0.88705205 0.88058824 0.88291876 0.88181081 0.87920948 0.89492601]

################################ PolynomialFeatures ################################
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2)
xp = pf.fit_transform(x)
print(xp.shape) #(581012, 1540) (증폭)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size = 0.1, random_state = 66)

model = make_pipeline(StandardScaler(), RandomForestClassifier())

model.fit(x_train, y_train)

print(round(model.score(x_test, y_test),4)) 
# 0.9596

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv = 7, scoring = 'r2')
print(scores) 
''' 
###### 기존 ######
기존 model.score) 0.9723070832938908
컬럼 제거 후 model.score) 0.9724619846303452

###### PolynomialFeatures ######
0.9596
'''