import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures  

x = np.arange(8).reshape(4,2)

print(x)
''' 
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
'''
print(x.shape) # (4, 2)

pf = PolynomialFeatures(degree = 2) # degree = 2 이면 이차방정식.

xp = pf.fit_transform(x)
print(xp) 
'''  
[[ 1.  0.  1.  0.  0.  1.]    # 1 + x1 + x2 + (x1)2 + x1x2 + (x2)2
 [ 1.  2.  3.  4.  6.  9.]
 [ 1.  4.  5. 16. 20. 25.]
 [ 1.  6.  7. 36. 42. 49.]]
'''
print(xp.shape)  # (4, 6)

##################################################################################

x = np.arange(12).reshape(4,3)

print(x)
''' 
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
'''
print(x.shape) # (4, 3)

pf = PolynomialFeatures(degree = 2) # degree = 2 이면 이차방정식.

xp = pf.fit_transform(x)
print(xp) 
'''  
[[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
 [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
 [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
 [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
'''
print(xp.shape)  # (4, 10)

##################################################################################

x = np.arange(8).reshape(4,2)

print(x)
''' 
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
'''
print(x.shape) # (4, 2)

pf = PolynomialFeatures(degree = 3) # degree = 3 이면 삼차방정식.

xp = pf.fit_transform(x)
print(xp) 
'''  
[[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
 [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
 [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
 [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]
'''
print(xp.shape)  # (4, 10)