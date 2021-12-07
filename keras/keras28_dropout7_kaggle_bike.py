from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input,Dropout
import pandas as pd  
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint


def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test,y_pred)) 


#1)데이터
path = "../_data/kaggle/bike/"      #.지금 현재 작업 공간   / ..이전


train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
#print(x.columns)

test_file = test_file.drop(['datetime'], axis=1)

#print(x.shape)    # (10886, 8)

y = train['count']
#print(y.shape)  #(10886,)

#로그변환
y = np.log1p(y)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움
test_file = scaler.transform(test_file)


#2)모델


input1 = Input(shape=(8,))  
dense1 = Dense(50,activation = 'relu')(input1)  
dense2 = Dense(40)(dense1)   
dense3 = Dense(30)(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(30)(drop1)
dense5 = Dense(30)(dense4)
drop2= Dropout(0.1)(dense5)
output1 = Dense(1)(drop2)
model = Model(inputs=input1, outputs=output1) 




#model = load_model("./_save/keras_07.1_kaggle_bike_save_model.h5")

#model.summary()

#3)컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

# #####################################################################################
# import datetime
# date = datetime.datetime.now()   #현재시간
# datetime = date.strftime("%m%d_%H%M")  #string형태로 만들어랏!  %m%d_%H%M = 월일/시/분    #1206_0456
# #print(datetime)

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #04d: 4자리까지(ex:9999),,, 4f: 소수 4제자리까지빼라     # hish에서 반환한 값이 epoch랑 val_loss로 들어가는것
# model_path = "".join([filepath, 'k27_' , datetime, '_', filename])  #" 구분자가 들어갑니다. ".join  <---

# #   ./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf5

# ########################################################################################


es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras28_7_MCP.hdf5') 



model.fit(x_train, y_train, epochs=10000, validation_split=0.2,callbacks=[es,mcp]) 

model.save("./_save/keras_028.7_kaggle_bike_save_model.h5")


#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

rmse = RMSE(y_test, y_pred)
print("RMSE: ", rmse)  




###================================================== 제출용
results = model.predict(test_file)

submit_file['count'] = results

#print(submit_file[:10])

submit_file.to_csv( path + "submitfile.csv", index=False)






print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras_028.7_kaggle_bike_save_model.h5')  

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)


print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('./_ModelCheckPoint/keras28_7_MCP.hdf5')  

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)

"""
    
 loss:  1.3878892660140991
r2:  0.29173234840228934
RMSE:  1.1780870206097986
============================1. 기본 출력=======================
69/69 [==============================] - 0s 455us/step - loss: 1.3879
loss:  1.3878892660140991
r2스코어 :  0.29173234840228934
============================2. load_model 출력=======================
69/69 [==============================] - 0s 440us/step - loss: 1.3879
loss2:  1.3878892660140991
r2스코어 :  0.29173234840228934
============================3. ModelCheckPoint 출력=======================
69/69 [==============================] - 0s 460us/step - loss: 1.3597
loss3:  1.3596904277801514
r2스코어 :  0.3061225771895463  


dropout 적용후

============================1. 기본 출력=======================
69/69 [==============================] - 0s 426us/step - loss: 1.3520
loss:  1.3519655466079712
r2스코어 :  0.31006468952772304
============================2. load_model 출력=======================
69/69 [==============================] - 0s 458us/step - loss: 1.3520
loss2:  1.3519655466079712
r2스코어 :  0.31006468952772304
============================3. ModelCheckPoint 출력=======================
69/69 [==============================] - 0s 411us/step - loss: 1.3469
loss3:  1.3469008207321167
r2스코어 :  0.312649400251401
      
"""

