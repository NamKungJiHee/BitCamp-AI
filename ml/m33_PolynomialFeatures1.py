from random import random
import sklearn
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datasets = load_boston()
#datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)   # 판다스에서는 describe()
print(x.shape, y.shape) # (506, 13) (506,) boston 

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 66)

#model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(round(model.score(x_test, y_test),4)) 
# 0.6278071803149758  r2_score  그냥
# 0.6278071803149635   pipeline

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv = 7, scoring = 'r2')
print(scores)
# [0.60874842 0.58740602 0.5951474  0.59874774 0.58950775 0.59927755 0.61555961]

# import sklearn
# print(sklearn.metrics.SCORERS.keys())  # 32번째 줄에서 scoring keys로 쓸만한 목록들 출력

################################ PolynomialFeatures ################################
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2)
xp = pf.fit_transform(x)
print(xp.shape) # (506, 105) = boston # x제곱 + x1x2... 이런식으로 해서 13의 y를 105개의 y로 늘려준다. (증폭의 개념)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size = 0.1, random_state = 66)

#model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(round(model.score(x_test, y_test),4)) 
# 0.9383

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv = 7, scoring = 'r2')
print(scores) 