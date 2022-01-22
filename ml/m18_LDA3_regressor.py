import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')

### LDA는 소수점 계산을 못함 ###

#1. 데이터
#datasets = load_boston()
#datasets = load_diabetes()
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

y = np.round(y,0) # 가장 가까운 정수로 반올림
print(y)

print('LDA 전:',x.shape)

a = []
for i in y:
    a.append(len(str(i).split('.')[1]))
print(np.unique(a,return_counts=True))

print(a)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis() 
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
LDA 후: (404, 13)
results:  0.9016514433565562
걸린 시간:  0.17455005645751953

2. diabetes
LDA 전: (442, 10)
LDA 후: (353, 10)
results:  0.313354229055848
걸린 시간:  0.15268707275390625

3. fetch_california_housing
LDA 전: (20640, 8)
LDA 후: (16512, 5)
results:  0.6575931583547636
걸린 시간:  1.0518989562988281
"""