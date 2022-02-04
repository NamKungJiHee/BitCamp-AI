from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, r2_score
# 비지도 학습: y가 없는 것 / 비지도학습으로 y를 찾아내는 것!  ## 분류에서만 쓸 수 있다!! ##

datasets = load_iris()
#x = datasets.data
# y = datasets.target
irisDF = pd.DataFrame(datasets.data, columns = [datasets.feature_names])  # 판다스 형식으로 바꿔준것!
#print(type(x))  # <class 'pandas.core.frame.DataFrame'>
#print(irisDF)

kmeans = KMeans(n_clusters = 3, random_state=66) #구간을 3개로 잡겠다!
kmeans.fit(irisDF)

#print(kmeans.labels_)   # y의 target값을 비지도로 뽑아낸것! 
''' 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
 2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
 2 1]
'''
#print(datasets.target)  # 실제 y의 target 값 (kmeans의 결과와 거의 유사)
''' 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
irisDF['cluster'] = kmeans.labels_
irisDF['target'] = datasets.target

print(accuracy_score(datasets.target, kmeans.labels_))
# 0.8933333333333333