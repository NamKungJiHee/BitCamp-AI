import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action = "ignore")

#1. 데이터
#datasets = load_boston()
datasets = load_breast_cancer()
#datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape) # (506, 13) -> (20640, 8)  # boston
# print(x.shape) # (569, 30) # cancer

pca = PCA(n_components = 14) # 컬럼 13개를 8개로 줄여주겠다. # 차원 축소라는 것은 y를 건들이는 것이 아니다.(x만 건들인다)
x = pca.fit_transform(x)
#print(x)
#print(x.shape) # (506, 5) 
print(x.shape) # (569, 14)

pca_EVR = pca.explained_variance_ratio_  # 설명가능한 변화률
print(pca_EVR)
print(sum(pca_EVR)) # 0.9999998946838411
''' 
[9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
 8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
 3.44135279e-07 1.86018721e-07] = 0.982, 0.016..
'''
cumsum = np.cumsum(pca_EVR)   #####성능#####
print(cumsum) # 누적합 
'''  누적합 30으로 뽑았을 때) 압축을 시켜주어도 1은 성능을 다 낼 수 있으니 제외시켜줘도 된다.
[0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.        ]
'''

import matplotlib.pyplot as plt
plt.plot(cumsum)
#plt.plot(pca_EVR)
plt.grid() # 격자
plt.show()