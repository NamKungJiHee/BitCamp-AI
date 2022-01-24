import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan , 8, 10], [2, 4, np.nan, 8, np.nan], 
                     [np.nan, 4, np.nan, 8, 10], [np.nan, 4, np.nan, 8, np.nan]])  # DataFrame = 행렬

#print(data.shape) # (4, 5)
data = data.transpose() 
data.columns = ['a', 'b', 'c', 'd'] # 컬럼명 추가해줌
print(data)
# print(data.shape) # (5, 4)

# 결측치 확인
#print(data.isnull())  # True가 nan
''' 
       a      b      c      d
0  False  False   True   True
1   True  False  False  False
2   True   True   True   True
3  False  False  False  False
4  False   True  False   True
'''
#print(data.isnull().sum())  # 결측치 갯수 보게 하기
''' 
a    2
b    2
c    2
d    3
'''
#print(data.info())
''' 
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   a       3 non-null      float64
 1   b       3 non-null      float64
 2   c       3 non-null      float64
 3   d       2 non-null      float64
'''

#1. 결측치 삭제
#print(data.dropna())
#print(data.dropna(axis = 0)) # nan값들있는 행 삭제 / axis = 0 (행)
''' 
     a    b    c    d
3  8.0  8.0  8.0  8.0
'''
#print(data.dropna(axis = 1))  # Empty DataFrame // axis = 1 (열)

#2-1. 특정값 - 평균값
# means = data.mean()
# #print(means)  # 평균값
# data2 = data.fillna(means)  # nan자리에 평균값으로 채워준다
# print(data2)
''' 
a    6.666667
b    4.666667
c    7.333333
d    6.000000
'''

#2-2. 특정값 - 중위값(중간값)
# meds = data.median()
# #print(meds)
# data2 = data.fillna(meds)
# print(data2)
''' 
a    8.0
b    4.0
c    8.0
d    6.0
'''
#2-3. 특정값 - ffill, bfill
#data2 = data.fillna(method = 'ffill')
#print(data2)
''' 
      a    b     c    d
0   2.0  2.0   NaN  NaN   
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
      a    b     c    d
0   2.0  2.0   NaN  NaN   ## 위의 값이 없기 때문에 그대로 NAN이다.
1   2.0  4.0   4.0  4.0
2   2.0  4.0   4.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0
########### 위에 값으로 nan값이 채워진다.
'''
# data2 = data.fillna(method = 'bfill')
# print(data2)
''' 
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
      a    b     c    d
0   2.0  2.0   4.0  4.0
1   8.0  4.0   4.0  4.0
2   8.0  8.0   8.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN  ## 뒤에 값이 없어서 그대로NAN
'''

# data2 = data.fillna(method = 'ffill', limit = 1)  # limit = 1은 하나만 그대로 가져오겠다는 것!
# print(data2)
''' 
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   2.0  4.0   4.0  4.0
2   NaN  4.0   4.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0

'''
# data2 = data.fillna(method = 'bfill', limit = 1)
# print(data2)
''' 
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
==========================
      a    b     c    d
0   2.0  2.0   4.0  4.0
1   NaN  4.0   4.0  4.0
2   8.0  8.0   8.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
#2-3. 특정값 - 채우기
data2 = data.fillna(747474)
print(data2)
''' 
          a         b         c         d
0       2.0       2.0  747474.0  747474.0
1  747474.0       4.0       4.0       4.0
2  747474.0  747474.0  747474.0  747474.0
3       8.0       8.0       8.0       8.0
4      10.0  747474.0      10.0  747474.0
'''
##########################################특정 칼럼만!###########################################################
##### 컬럼별로 결측치 채워주는 기준이 달라야한다 #####

means = data['a'].mean()     # 'a'컬럼은 평균값으로 채워진다!
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()    # 'b'컬럼은 중위값으로 채워진다!
print(meds)
data['b'] = data['b'].fillna(meds)
print(data)