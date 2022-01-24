## 결측치 처리 ##
#1. 행 또는 열 삭제(Drop)
#2. 임의의 값
# fillna - 0, ffill(전위값-위), bfill(후위값-아래), 중위값, 평균값...
#3. 보간 - interpolate (선형보간법)
#4. 모델링 - predict
#5. 부스팅계열 - 통상 결측치, 이상치에 대해 자유로움

import pandas as pd
from datetime import datetime
import numpy as np

dates = ['1/24/2022', '1/25/2022', '1/26/2022', '1/27/2022','1/28/2022']
dates = pd.to_datetime(dates) 
print(dates)
''' 
DatetimeIndex(['2022-01-24', '2022-01-25', '2022-01-26', '2022-01-27',
               '2022-01-28'],)
'''
ts = pd.Series([2, np.nan, np.nan, 8, 10], index = dates)  # Series = 벡터 // DataFrame = 행렬
print(ts)
''' 
2022-01-24     2.0
2022-01-25     NaN
2022-01-26     NaN
2022-01-27     8.0
2022-01-28    10.0
'''
ts = ts.interpolate() # 선형보간법(결측치) // Nan값 채워주기
print(ts)
''' 
2022-01-24     2.0
2022-01-25     4.0
2022-01-26     6.0
2022-01-27     8.0
2022-01-28    10.0
'''
