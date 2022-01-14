import numpy as np, pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time 

############# 화욜: 거래량 / 수욜: 시가###################

def split_xy5(dataset, time_steps, y_column):  
    x,y = list(), list()  # x와 y에 담겠다.
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    # 몇개(며칠치)로 나누어 줄것인가 
        y_end_number = x_end_number + y_column    # y값 의미(위치)  
        
        if y_end_number > len(dataset):   # 계속 반복하되 우리가 뽑으려는 결과값이 나오면 break하겠다.
            break
        
        tmp_x = dataset[i:x_end_number, :]  # 4개 행, 모든열(slice할 범위)  #ex)0~4 즉 3까지의 x컬럼을 [:]모두 범위로 잡겠다.
        tmp_y = dataset[x_end_number: y_end_number, 1]  # 3이니까 자릿값 0,1,2 (y값 자리)
        x.append(tmp_x)
        y.append(tmp_y)   
        
    return np.array(x),np.array(y)

path = "../_data/주식/"

s = pd.read_csv(path + '삼성전자.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:70,:].sort_values(['일자'],ascending=[True])
k = pd.read_csv(path + '키움증권.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:70,:].sort_values(['일자'],ascending=[True])   
    
#1) 데이터
s1 = s[['등락률', '거래량', '금액(백만)', '외인(수량)']].values 
k1 = k[['등락률', '거래량', '금액(백만)', '외인(수량)']].values  

# print(s1.shape) # (100, 4)
# print(k1.shape) # (100, 4)

x1, y1 = split_xy5(s1,4,3)  # 월,화 거래량 데이터 뽑아내기
x2, y2 = split_xy5(k1,4,3)

# print(x1.shape)  #  (45, 4, 4)
# print(y1.shape)  #  (45, 2)
# print(x2.shape)  #  (45, 4, 4)
# print(y2.shape)  #  (45, 2)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2 ,train_size=0.8,random_state=66)

# print(x1_train.shape,x1_test.shape)  # (36, 4, 4) (9, 4, 4)
# print(y1_train.shape, y1_test.shape)  # (36, 2) (9, 2)
# print(x2_train.shape,x2_test.shape)  # (36, 4, 4) (9, 4, 4)
# print(y2_train.shape, y2_test.shape)  # (36, 2) (9, 2)


#2) 모델링
#######s1 데이터##########
input1 = Input(shape= (4,4))
dense1 = LSTM(10, activation = 'relu', name = 'dense1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'dense2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'dense3')(dense2)
output1 = Dense(30, activation = 'relu', name = 'output1')(dense3)


###### k1 데이터 ##########
input2 = Input(shape= (4,4))
dense01 = LSTM(15, activation = 'relu', name = 'dense01')(input2)
dense02 = Dense(20, activation = 'relu', name = 'dense02')(dense01)
dense03 = Dense(20, activation = 'relu', name = 'dense03')(dense02)
output2 = Dense(30, name = 'output2')(dense03)

from tensorflow.keras.layers import concatenate,Concatenate 

merge1 = concatenate([output1,output2])

#output 모델1
output01 = Dense(15,activation='relu')(merge1)
output02 = Dense(25,activation='relu')(output01)
output03 = Dense(30,activation='relu')(output02)
last_output1 = Dense(2)(output03) 

#output 모델2
output11 = Dense(5,activation='relu')(merge1)
output12 = Dense(5)(output11)
output13 = Dense(5)(output12)
output14 = Dense(40,activation='relu')(output13)
last_output2 = Dense(2)(output14)  

model = Model(inputs=[input1, input2], outputs = [last_output1,last_output2])

#3) 컴파일, 훈련
model.compile(loss='mae', optimizer= 'adam')

es = EarlyStopping(monitor='val_loss', verbose=1, mode = 'auto', patience=30)

model.fit([x1_train, x2_train], [y1_train, y2_train], batch_size=20, validation_split=0.2, epochs=1000, callbacks=[es])


model.save("../_data/주식/keras.거래량.h5")
#model = load_model('../_data/주식/keras.거래량.h5')

#4) 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test])
print('loss: ', loss)

y_predict1, y_predict2 = model.predict([x1,x2])

# for i in range(5):
#     print('종가: ', [y1_test[i],y2_test[i]], '예측가: ', y_predict2[i])

print("삼성 거래량: ", y_predict1[-1][-1]) 
print("키움 거래량: ", y_predict2[-1][-1]) 



"""
loss:  [3750758.5, 3732916.25, 17842.2578125]
삼성 거래량:  10992882.0
키움 거래량:  48753.035
"""
