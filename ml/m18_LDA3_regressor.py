import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')

### LDA는 소수점 계산을 못함 ### 
### LDA에서 분류에서는 디폴트값으로 차원을 줄여주지만 회귀에서는 이것이 안먹힌다. (디폴트값: y label갯수-1) // 그래서 회귀에서는 직접 n_component를 해줘야함 ###

#1. 데이터
#datasets = load_boston()
#datasets = load_diabetes()
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

y = np.round(y,0) # 가장 가까운 정수로 반올림
#print(y)

print('LDA 전:',x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis(n_components=3) # n_components
# x = pca.fit_transform(x)         

lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print('LDA 후:',x_train.shape)

#2. 모델
from xgboost import XGBClassifier, XGBRegressor
model = XGBRegressor()

#3. 훈련
import time
start = time.time()
# model.fit(x_train, y_train, eval_metric='error')
#model.fit(x_train, y_train, eval_metric='rmse')
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('results: ', results)
print('걸린 시간: ', end-start)

""" 
1. boston
LDA 전: (506, 13)
LDA 후: (404, 3) 
results:  0.8196061188057238   
걸린 시간:  0.15267276763916016

2. diabetes
LDA 전: (442, 10)
LDA 후: (353, 3)
results:  0.3999594042985677
걸린 시간:  0.15825533866882324

3. fetch_california_housing
LDA 전: (20640, 8)
LDA 후: (16512, 3)
results:  0.6362060739925103
걸린 시간:  1.0233378410339355
"""