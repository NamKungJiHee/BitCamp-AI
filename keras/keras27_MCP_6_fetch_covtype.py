from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)   # (581012, 54) (581012,) 
#print(np.unique(y))

import pandas as pd   
y = pd.get_dummies(y)   
# print(y.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)  
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2) 모델링

input1 = Input(shape=(54,))  
dense1 = Dense(10,activation = 'relu')(input1)  
dense2 = Dense(10)(dense1)   
dense3 = Dense(10,activation = 'relu')(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(7,activation = 'softmax')(dense4)
model = Model(inputs=input1, outputs=output1) 


#model = load_model("./_save/keras_06.1_fetch_covtype_save_model.h5")

#model.summary()


#3) 캄파일, 훈련

model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

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


es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras27_6_MCP.hdf5') 


model.fit(x_train, y_train, epochs = 10000, batch_size=500, validation_split= 0.2,callbacks=[es,mcp])  

model.save("./_save/keras_06.1_fetch_covtype_save_model.h5")


#4) 평가, 예측

print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras_06.1_fetch_covtype_save_model.h5')  

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)


print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('./_ModelCheckPoint/keras27_6_MCP.hdf5')  

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)

"""
  
loss:  [0.5203027725219727, 0.7824496626853943]
r2스코어 :  0.4794340532262525
============================2. load_model 출력=======================
3632/3632 [==============================] - 1s 391us/step - loss: 0.5203 - accuracy: 0.7824
loss2:  [0.5203027725219727, 0.7824496626853943]
r2스코어 :  0.4794340532262525
============================3. ModelCheckPoint 출력=======================
3632/3632 [==============================] - 1s 382us/step - loss: 0.5203 - accuracy: 0.7824
loss3:  [0.5203027725219727, 0.7824496626853943]
r2스코어 :  0.4794340532262525
    
    
"""