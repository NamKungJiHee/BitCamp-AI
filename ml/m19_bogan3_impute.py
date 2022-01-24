import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan , 8, 10], [2, 4, np.nan, 8, np.nan], 
                     [np.nan, 4, np.nan, 8, 10], [np.nan, 4, np.nan, 8, np.nan]])  # DataFrame = 행렬

#print(data.shape) # (4, 5)
data = data.transpose() 
data.columns = ['a', 'b', 'c', 'd'] # 컬럼명 추가해줌
# print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#imputer = SimpleImputer(strategy='mean') # 평균값으로 결측치 채워주기
#imputer = SimpleImputer(strategy='median') # 중위값으로 결측치 채워주기
#imputer = SimpleImputer(strategy='most_frequent') # 제일 많이 사용한 값으로 결측치 채워주기
imputer = SimpleImputer(strategy='constant', fill_value=777) # 'constant'만 쓰면 0으로 결측치 채워주고 fill_value = 777을 쓰면 nan자리에 그 값이 들어감

### EX) imputer = SimpleImputer(strategy='mean', fill_value=777) 이렇게 쓰면 fill_value값이 우선적으로 적용된다. ###

'''
means = data['a'].mean()   
data['a'] = data['a'].fillna(means)
''' 

# fit에는 dataframe이 들어가는데, 우리는 컬럼만 바꾸고 싶다.
# 시리즈를 넣으면 에러가 난다.
# 처리!!

data2 = data.copy()

# 1개만 할 경우
# data[['a']] = imputer.fit_transform(data[['a']])
# print(data)

# 2개이상 할 경우
data2[['a','c']] = imputer.fit_transform(data[['a','c']])
print(data2)