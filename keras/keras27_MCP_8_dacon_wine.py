import numpy as np
import pandas as pd  
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input
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


#print(type(train))         
#print(train.info())    
#print(train.describe())  

##dtypes: float64(11), int64(2), object(1)


#print(train.columns)  # x--> Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
      # 'residual sugar', 'chlorides', 'free sulfur dioxide',
      # 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
      # 'quality'],
     #  dtype='object')



#print(submit.columns)



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
print(y.shape)  

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
dense5 = Dense(40)(dense4)
dense6 = Dense(20)(dense5)
output1 = Dense(5, activation='softmax')(dense6)
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
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras27_8_MCP.hdf5') 

model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.25, callbacks=[es,mcp]) 

model.save("./_save/keras_08.1_dacon_wine_save_model.h5")
#model = load_model("./_save/keras_08.1_dacon_wine_save_model.h5")

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)




###================================================== 제출용


result = model.predict(test_file)
#print(result)
result_int = np.argmax(result, axis =1).reshape(-1,1) + 4 # 결과를 열로 뽑겠따!
submit_file['quality'] = result_int

# argmax: 원핫인코딩된 데이터를 결과데이터에 넣을때 다시 숫자로, 되돌려 주는 편리한 기능을 제공해주는 함수 / 확률을 다시 change
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv( path +f"result/accuracy_{acc}.csv", index=False)  # 디폴트: 기본으로 index가 생성됨 / if index= false하면 인덱스 생성x



print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras_08.1_dacon_wine_save_model.h5')  

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)


print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('./_ModelCheckPoint/keras27_8_MCP.hdf5')  

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)

"""
loss:  [1.0409561395645142, 0.5517774224281311]
r2스코어 :  0.01819808542979586
============================2. load_model 출력=======================
21/21 [==============================] - 0s 400us/step - loss: 1.0410 - accuracy: 0.5518
loss2:  [1.0409561395645142, 0.5517774224281311]
r2스코어 :  0.01819808542979586
============================3. ModelCheckPoint 출력=======================
21/21 [==============================] - 0s 527us/step - loss: 0.9588 - accuracy: 0.5935
loss3:  [0.9587953090667725, 0.5935084819793701]
r2스코어 :  0.1376288435803073     
      
"""