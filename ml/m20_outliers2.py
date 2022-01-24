import numpy as np,pandas as pd
aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],[100, 200, 3, 400, 500, 600, 700, 800, 900, 1000, 1001, 1002, 99]])
# (2, 13) --> (13, 2)
aaa = np.transpose(aaa)  # (13, 2)

#다차원의 outlier가 출력되도록 수정!!

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

# case 1.
# aaa = pd.DataFrame(aaa,columns=('x','y'))
# a = aaa['x'].values
# b = aaa['y'].values

# case 2.
a = []
b = []
for i in aaa:
    a.append(i[0])
    b.append(i[1])

print(aaa) 
print(a)  
print(b) 
print(outliers(b))

''' 
[[   1  100]
 [   2  200]
 [ -20    3]
 [   4  400]
 [   5  500]
 [   6  600]
 [   7  700]
 [   8  800]
 [  30  900]
 [ 100 1000]
 [ 500 1001]
 [  12 1002]
 [  13   99]]
[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13]
[100, 200, 3, 400, 500, 600, 700, 800, 900, 1000, 1001, 1002, 99]
1사분위:  200.0
q2:  600.0
3사분위:  900.0
iqr:  700.0
'''