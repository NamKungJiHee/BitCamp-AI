import numpy as np
import pandas as pd  
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1)데이터

path = "../_data/dacon/wine/"      #.지금 현재 작업 공간   / ..이전

train = pd.read_csv(path + 'train.csv')
#print(train)   #[3231 rows x 14 columns]
test_file = pd.read_csv(path + 'test.csv')  #[3231 rows x 13 columns]
#print(test_file)
submit_file = pd.read_csv(path + 'sample_submission.csv')
#print(submit_file)  #[3231 rows x 2 columns

x = train.drop(['id','quality'], axis=1) 
#print(x.columns)

test_file = test_file.drop(['id'], axis=1)

#print(x.shape)    # (3231, 12)

y = train['quality']
#print(y.shape)  # (3231,)

le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

from pandas import get_dummies 
y = get_dummies(y)
print(y.shape)  # (3231, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train) 

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2)모델

input1 = Input(shape=(12,))
dense1 = Dense(40)(input1)
dense2 = Dense(60, activation = 'relu')(dense1)
dense3 = Dense(80, activation = 'relu')(dense2)
dense4 = Dense(60)(dense3)
drop1=Dropout(0.2)(dense4)
dense5 = Dense(40)(drop1)
dense6 = Dense(20)(dense5)
drop2= Dropout(0.3)(dense6)
output1 = Dense(5, activation='softmax')(drop2)
model = Model(inputs=input1, outputs=output1)


#3)컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

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
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=False, filepath = './_ModelCheckPoint/keras28_5_MCP.hdf5') 


model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.25, callbacks=[es,mcp]) 
model.save("./_save/keras_028.5_wine_save_model.h5")

#model = load_model("./_save/keras_028.5_wine_save_model.h5")


#4)평가, 예측
print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras_028.5_wine_save_model.h5')  

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)


print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('./_ModelCheckPoint/keras28_5_MCP.hdf5')  

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)


###================================================== 제출용


result = model.predict(test_file)
#print(result)
result_int = np.argmax(result, axis =1).reshape(-1,1) + 4 # 결과를 열로 뽑겠따!
submit_file['quality'] = result_int

# argmax: 원핫인코딩된 데이터를 결과데이터에 넣을때 다시 숫자로, 되돌려 주는 편리한 기능을 제공해주는 함수 / 확률을 다시 change
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv( path +f"result/accuracy_{acc}.csv", index=False)  # 디폴트: 기본으로 index가 생성됨 / if index= false하면 인덱스 생성x

"""
loss:  [0.987700343132019, 0.5765069723129272]
r2스코어 :  0.11464358633334415
============================2. load_model 출력=======================
21/21 [==============================] - 0s 494us/step - loss: 0.9877 - accuracy: 0.5765
loss2:  [0.987700343132019, 0.5765069723129272]
r2스코어 :  0.11464358633334415
============================3. ModelCheckPoint 출력=======================
21/21 [==============================] - 0s 399us/step - loss: 1.0028 - accuracy: 0.5672
loss3:  [1.0028051137924194, 0.5672333836555481]
r2스코어 :  0.10381701057145507  


dropout 적용시

============================1. 기본 출력=======================
21/21 [==============================] - 0s 450us/step - loss: 0.9753 - accuracy: 0.5765
loss:  [0.9752734303474426, 0.5765069723129272]
r2스코어 :  0.12164582559432675
============================2. load_model 출력=======================
21/21 [==============================] - 0s 499us/step - loss: 0.9753 - accuracy: 0.5765
loss2:  [0.9752734303474426, 0.5765069723129272]
r2스코어 :  0.12164582559432675
============================3. ModelCheckPoint 출력=======================
21/21 [==============================] - 0s 407us/step - loss: 0.9753 - accuracy: 0.5765
loss3:  [0.9752734303474426, 0.5765069723129272]
r2스코어 :  0.12164582559432675







"""