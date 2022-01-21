import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')

# LDA) y의 target값을 넣어서 전처리 해준다! lable값에 맞추어 차원 축소

#1. 데이터
#datasets = load_iris()
#datasets = load_breast_cancer()
#datasets = load_wine()
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print("LDA 전 : ", x.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)

# stratify: 균등하게 빼주는 것

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

print("LDA 후: ", x_train.shape)

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
import time
start = time.time()
#model.fit(x_train, y_train, eval_metric='error') # 이진분류
model.fit(x_train, y_train, eval_metric='merror')  # 다중분류
# model.fit(x_train, y_train)

end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('results:', results)
print('걸린 시간: ', end-start)

''' 
1) iris
LDA 전 :  (150, 4)
LDA 후:  (120, 2)
results: 1.0
걸린 시간:  0.10931849479675293

2) cancer
LDA 전 :  (569, 30)
LDA 후:  (455, 1)
results: 0.9473684210526315
걸린 시간:  0.08281254768371582

3) wine
LDA 전 :  (178, 13)
LDA 후:  (142, 2)
results: 1.0
걸린 시간:  0.10841250419616699

4) fetch_covtype
LDA 전 :  (581012, 54)
LDA 후:  (464809, 6)
results: 0.7878109859470065
걸린 시간:  155.59642100334167
'''