# import numpy as np
# aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13], [100, 200, 3, 400, 500, 600, 700, 800, 900, 1000, 1001, 1002, 99]])
# # (2, 13) --> (13, 2)
# aaa = np.transpose(aaa)  # (13, 2)

# from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=0.3)   

# outliers.fit(aaa)
# results = outliers.predict(aaa)
# print(results)

'''   
##### contamination: 오염도 = outliers #####

범위 밖에서부터 오염도를 나타내준다!

contamination = 0.1
[ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]
-1: 100, 500

contamination = 0.2
[ 1  1  1  1  1  1  1  1 -1 -1 -1  1  1]

contamination=0.5
[-1  1 -1  1  1  1  1  1 -1 -1 -1 -1 -1] 
'''

import numpy as np
import pandas as pd

aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)

# df = pd.DataFrame(aaa, columns=['x','y'])

# data1 = df[['x']]
# data2 = df[['y']]

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)
pred = outliers.fit_predict(aaa)
print(pred.shape) # (13,)

b = list(pred)
print(b.count(-1))
index_for_outlier = np.where(pred == -1)
print('outier indexex are', index_for_outlier)
outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value)

''' 
(13,)
4
outier indexex are (array([ 2,  8,  9, 10], dtype=int64),)
outlier_value : [[ -20    3]
 [  30  900]
 [ 100  190]
 [ 500 1001]]
'''