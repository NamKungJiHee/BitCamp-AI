from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, r2_score
# 비지도 학습: y가 없는 것 / 비지도학습으로 y를 찾아내는 것!  ## 분류에서만 쓸 수 있다!! ##

datasets = load_wine()
#x = datasets.data
# y = datasets.target
wineDF = pd.DataFrame(datasets.data, columns = [datasets.feature_names])  # 판다스 형식으로 바꿔준것!
#print(type(x))  # <class 'pandas.core.frame.DataFrame'>
#print(wineDF)

kmeans = KMeans(n_clusters = 3, random_state=66) 
kmeans.fit(wineDF)

print(kmeans.labels_)   # y의 target값을 비지도로 뽑아낸것! 
''' 
[1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
 1 1 0 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 0 2 0 2 2 0 2 2 0 0 0 2 2 1
 0 2 2 2 0 2 2 0 0 2 2 2 2 2 0 0 2 2 2 2 2 0 0 2 0 2 0 2 2 2 0 2 2 2 2 0 2
 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 2 2 0 0 0 0 2 2 2 0 0 2 2 0 0 2 0
 0 2 2 2 2 0 0 0 2 0 0 0 2 0 2 0 0 2 0 0 0 0 2 2 0 0 0 0 0 2]
'''
print(datasets.target)  # 실제 y의 target 값
''' 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
wineDF['cluster'] = kmeans.labels_
wineDF['target'] = datasets.target

#print(accuracy_score(datasets.target, kmeans.labels_))
# 0.1853932584269663