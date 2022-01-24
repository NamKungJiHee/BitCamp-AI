import numpy as np
aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500, 12, 13])
#               0  1   2     3  4  5  6  7  8   9     10  11  12
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])  # percentile : 백분위수  
                                                                       
    print("1사분위: ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3 - quartile_1   # 통상적으로 3사분위에서 1사분위를 빼고 그 값에 1.5를 곱해주기 -> 1,3사분위 위치에서 각각 그 값을 더해주거나 빼줌
    print("iqr: ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)
print("이상치의 위치: ", outliers_loc)
''' 
1사분위:  4.0
q2:  7.0
3사분위:  13.0
iqr:  9.0
이상치의 위치:  (array([ 2,  8,  9, 10], dtype=int64),)   --> index 위치 == -1000, 90, 100, 500
'''
""" 
return np.where((data_out>upper_bound) | (data_out<lower_bound))
                                     또는         data_out에 있는 것들 : [2,  8,  9, 10]    
                                          
                                          
## np.where() = 위치를 배열로 반환 ##                                          
"""
#시각화
#boxplot으로 그리기!
import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()  # 동그라미: 이상치
