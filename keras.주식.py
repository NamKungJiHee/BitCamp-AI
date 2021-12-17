import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = "../_data/주식/"

s = pd.read_csv(path + '삼성전자.csv', header=0)
k = pd.read_csv(path + '키움증권.csv')

s1 = pd.DataFrame(s)
#print(s1.shape) # (1120, 17)
# print(type(s1))    # <class 'pandas.core.frame.DataFrame'>

k1 = pd.DataFrame(k)
#print(k1.shape) # (1060, 17)
# print(type(k1))    # <class 'pandas.core.frame.DataFrame'>

#print(s.info())     # float64(3), object(14)
#print(s.describe()) 
#print(k.info())  # dtypes: float64(3), object(14)


# x = s['시가','고가','저가','종가','거래량'] #외인, 외인비
# y = s['종가']

x = s.drop(['시가'],axis=1)  
x = x.to_numpy()    


#print(x)

# le = LabelEncoder()
# le.fit(x['type'])
# x['type'] = le.transform(x['type'])

# le.fit(test_file['type'])
# test_file['type'] = le.transform(test_file['type'])






