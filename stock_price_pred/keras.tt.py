import numpy as np, pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_xy5(dataset, time_steps, y_column):  
    x,y = list(), list()  # x와 y에 담겠다.
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    # 몇개(며칠치)로 나누어 줄것인가 
        y_end_number = x_end_number + y_column    # y값 의미(위치)  
        
        if y_end_number > len(dataset):   # 계속 반복하되 우리가 뽑으려는 결과값이 나오면 break하겠다.
            break
        
        tmp_x = dataset[i:x_end_number, 1:]  # 4개 행, 모든열(slice할 범위)  #ex)0~4 즉 3까지의 x컬럼을 [:]모두 범위로 잡겠다.
        tmp_y = dataset[x_end_number: y_end_number, 3]  # [a:b] : [가져올 행의 인덱스. 가져올 열의 인덱스]
        x.append(tmp_x)
        y.append(tmp_y)   
        
    return np.array(x),np.array(y)

path = "../_data/주식/"

s = pd.read_csv(path + '삼성전자.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:100,:].sort_values(['일자'],ascending=[True])
k = pd.read_csv(path + '키움증권.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:100,:].sort_values(['일자'],ascending=[True])   
    
#1) 데이터
s1 = s[['거래량','시가','저가','종가']].values 
k1 = k[['시가','고가','저가','종가']].values  


#print(s1[-4:]) # 삼성 뒤에서부터 4줄 출력

x1, y1 = split_xy5(s1,4,5)  
#x2, y2 = split_xy5(k1,4,3)

print(x1)
print(y1)  
#print(x1[:, 1:])  # (100으로 자르기 : 0~2까지의 값 출력, 1:했으므로 열을 1로 자르겠다.)  # tmp_x
#print(y1[:8,1])                                                                      # tmp_y

