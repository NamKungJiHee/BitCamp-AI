import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time 

############# 월: 종가 / 화욜: 거래량 / 수욜: 시가###################

def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  # x와 y에 담겠다.
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    # 몇개로 나누어 줄것인가
        y_end_number = x_end_number + y_column    # y값 의미
        
        if y_end_number > len(dataset):   # 계속 반복하되 우리가 뽑으려는 결과값이 나오면 break하겠다.
            break
        
        tmp_x = dataset[i:x_end_number, :]  # 4개 행, 모든열
        tmp_y = dataset[x_end_number: y_end_number, 3]  # 3이니까 자릿값 0,1,2
        x.append(tmp_x)
        y.append(tmp_y)   
        
    return np.array(x),np.array(y)

path = "./"

s = pd.read_csv(path + '삼성전자.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:893,:].sort_values(['일자'],ascending=[True])  # # 내림차순 : ascending=False
k = pd.read_csv(path + '키움증권.csv', encoding = 'ANSI', header=0, index_col=0, thousands=',').iloc[:893,:].sort_values(['일자'],ascending=[True])   # index_col=0 : '일자' 버림

                                                                                                                                               # sort_values['일자'] = in data: '일자'를 기준으로 행 순서를 바꿔줌     

# print(s) # [1120 rows x 16 columns]
# print(k) # [1060 rows x 16 columns]

s1 = s[['시가','고가','저가','종가']].values  # x1, x2 값 설정  # 거래량','외인(수량)'
k1 = k[['시가','고가','저가','종가','외인(수량)','거래량']].values  

# print(s1.shape) # (893, 4)
# print(k1.shape) # (893, 6)


# samsung = pd.DataFrame(s1)[-5:].values.reshape(1,5,4)
x1, y1 = split_xy5(s1,5,1)
x2, y2 = split_xy5(k1,5,1)


# print(x1[-3:])  # 뒤에서부터 3개..
# print(y1[-3:])

# # print(x1.shape)   # 뒤에서부터   # (888, 5, 4)  # 3차원으로 change
# # print(y1.shape)  # (888, 1)
# print(x2.shape)   #  (888, 5, 6)  # 3차원으로 change
# print(y2.shape)  # (888, 1)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, train_size=0.8, random_state= 66, shuffle=True)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, train_size=0.8, random_state= 66, shuffle=True)

# # print(x1_train.shape,x1_test.shape)  # (710, 5, 4) (178, 5, 4)
# # print(y1_train.shape, y1_test.shape)  # (710, 1) (178, 1)
# print(x2_train.shape,x2_test.shape)  # (710, 5, 6) (178, 5, 6)
# print(y2_train.shape, y2_test.shape)  # (710, 1) (178, 1)

x1_train = x1_train.reshape(len(x1_train),-1)
x1_test = x1_test.reshape(len(x1_test),-1)
x2_train = x2_train.reshape(len(x2_train),-1)
x2_test = x2_test.reshape(len(x2_test),-1)

# print(x1_train.shape, x1_test.shape) # (710, 20) (178, 20)
# print(y1_train.shape, y1_test.shape)  # (710, 1) (178, 1)
# print(x2_train.shape, x2_test.shape) # (710, 30) (178, 30)
# print(y2_train.shape, y2_test.shape)  # (710, 1) (178, 1)

scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = RobustScaler() #scaler = MaxAbsScaler()

x1_train = scaler.fit_transform(x1_train).reshape(len(x1_train),5,4) 
x1_test = scaler.transform(x1_test).reshape(len(x1_test),5,4)
x2_train = scaler.fit_transform(x2_train).reshape(len(x2_train),6,5) 
x2_test = scaler.transform(x2_test).reshape(len(x2_test),6,5)   
    
  
# print(x1_train.shape, x1_test.shape) # (710, 5, 4) (178, 5, 4)
# print(y1_train.shape, y1_test.shape)  #  (710, 1) (178, 1)
# print(x2_train.shape, x2_test.shape) # (710, 6, 5) (178, 6, 5)
# print(y2_train.shape, y2_test.shape)  #  (710, 1) (178, 1)

y1 = np.log1p(y1)  
y2 = np.log1p(y2)  

##############################################
#######s1 데이터##########
input1 = Input(shape= (5,4))
dense1 = LSTM(5, activation = 'relu', name = 'dense1')(input1)
dense2 = Dense(10, activation = 'relu', name = 'dense2')(dense1)
dense3 = Dense(20, activation = 'relu', name = 'dense3')(dense2)
output1 = Dense(10, activation = 'relu', name = 'output1')(dense3)


###### k1 데이터 ##########
input2 = Input(shape= (6,5))
dense01 = LSTM(10, activation = 'relu', name = 'dense01')(input2)
dense02 = Dense(20, activation = 'relu', name = 'dense02')(dense01)
dense03 = Dense(30, activation = 'relu', name = 'dense03')(dense02)
output2 = Dense(10, name = 'output2')(dense03)

from tensorflow.keras.layers import concatenate,Concatenate 

merge1 = concatenate([output1,output2])

#output 모델1
output01 = Dense(7)(merge1)
output02 = Dense(20)(output01)
output03 = Dense(30,activation='relu')(output02)
last_output1 = Dense(1)(output03)  # y1의 열의 갯수

#output 모델2
output11 = Dense(7)(merge1)
output12 = Dense(30)(output11)
output13 = Dense(30)(output12)
output14 = Dense(20,activation='relu')(output13)
last_output2 = Dense(1)(output14)  # y2의 열의 갯수

model = Model(inputs=[input1, input2], outputs = [last_output1,last_output2])


# ################################################
model.compile(loss='mse', optimizer= 'adam', metrics=['mae'])

start = time.time()

es = EarlyStopping(monitor='val_loss', verbose=1, mode = 'auto', patience=30)

model.fit([x1_train, x2_train], [y1_train, y2_train], batch_size=10, validation_split=0.2, epochs=1000, callbacks=[es])
# model.fit(x1_train, y1_train, batch_size=10, validation_split=0.2, epochs=1000, callbacks=[es])
# model.fit(x2_train, y2_train, batch_size=10, validation_split=0.2, epochs=1000, callbacks=[es])

end = time.time()-start
print("걸린시간: ", round(end,3), '초')

#model.save("./save/keras.주식.h5")
model = load_model('./save/keras.주식.h5')

# ##################################################s1
# loss = model.evaluate(x1_test, y1_test)
# print('loss: ', loss)

# y_predict = model.predict(x1_test)
# for i in range(5):
#     print('종가: ', y1_test[i], '예측가: ', y_predict[i])

# from sklearn.metrics import r2_score
# r2 = r2_score(y1_test, y_predict) 
# print('r2스코어 : ', r2)
######################################################k1
# loss = model.evaluate(x2_test, y2_test)
# print('loss: ', loss)

# y_predict1 = model.predict(x2_test)
# for i in range(5):
#     print('종가: ', y2_test[i], '예측가: ', y_predict1[i])

# from sklearn.metrics import r2_score
# r2 = r2_score(y2_test, y_predict1) 
# print('r2스코어 : ', r2)
#######################################################
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test])
print('loss: ', loss)

y_predict1, y_predict2 = model.predict([x1_test,x2_test])
# for i in range(5):
#     print('종가: ', [y1_test[i],y2_test[i]], '예측가: ', y_predict2[i])

print("삼성: ", y_predict1[-1])
print("키움: ", y_predict2[-1])


# y_predict = model.predict(samsung)
# print('삼성예측: ', y_predict)


# from sklearn.metrics import r2_score
# r2 = r2_score([y1_test,y2_test], [y_predict1,y_predict2]) 
# print('r2스코어 : ', r2)

"""
걸린시간:  33.062 초
loss:  [11490025.0, 2549270.25, 8940754.0, 1165.408935546875, 2127.341796875]
삼성:  [41918.07]
키움:  [73649.02]
"""



