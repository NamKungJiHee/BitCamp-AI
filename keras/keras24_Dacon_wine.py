
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# import pandas as pd
# import time
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


# #1 데이터
# path = "D:\\_data\\dacon\\wine\\" 
# train = pd.read_csv(path +"train.csv")
# test_file = pd.read_csv(path + "test.csv") 
# submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

# y = train['quality']
# x = train.drop(['id','quality'], axis =1)
# x = x.drop(['citric acid','sulphates'], axis =1) #,<-------- 상관관계

# le = LabelEncoder()
# le.fit(train['type'])
# x['type'] = le.transform(train['type'])

# import matplotlib.pyplot as plt
# import seaborn as sns

# # plt.figure(figsize=(10,10))
# # sns.heatmap(data= x.corr(), square=True, annot=True, cbar=True)
# # plt.show()

# y = np.array(y).reshape(-1,1)
# enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
# enc.fit(y)
# y = enc.transform(y).toarray()
# x = x.to_numpy()

# ###############
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#          train_size = 0.8, shuffle = True, random_state = 66) 

# scaler = RobustScaler()

# n = x_train.shape[0]
# # x_train_reshape = x_train.reshape(n,-1) #
# x_train_transe = scaler.fit_transform(x_train) #<------ 0.0 ~ 1.0
# #################################################################
# # print(x_train_transe.shape) # (2584,9)
# x_train = x_train_transe.reshape(n,2,5,1) 
# m = x_test.shape[0]
# x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,5,1)


# #2 모델

# model = Sequential() 
# model.add(Conv2D(30, kernel_size=(2,2),padding ='same',strides=1, 
#                  input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))
# model.add(MaxPooling2D())
# model.add(Conv2D(70,(2,2),padding ='same', activation='relu'))#<------------
# # model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(70,(2,2),padding ='same', activation='relu'))
# # model.add(MaxPooling2D())
# model.add(Flatten())
# # print("==========================================★")
# model.add(Dense(90))
# model.add(Dropout(0.1))
# model.add(Dense(110))
# #model.add(Dense(130))
# model.add(Dropout(0.3))
# model.add(Dense(y.shape[1], activation = 'softmax')) 

# #3. 컴파일, 훈련
# opt = 'Nadam'     #'Adadelta' #'Nadam' 'adam'  
# epoch = 1000
# patience_num = 30


# model.compile(loss = 'categorical_crossentropy', optimizer = opt , metrics=['accuracy'])
# ########################################################################
# # model.compile(loss = 'mse', optimizer = 'adam')
# # start = time.time()
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# # date = datetime.datetime.now()
# # datetime = date.strftime("%m%d_%H%M")
# # filepath = "./_ModelCheckPoint/"
# # filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
# # model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
# es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
# # mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
# hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es], batch_size = 50)
# # end = time.time() - start
# # print('시간 : ', round(end,2) ,'초')
# ########################################################################
# test_file['type'] = le.transform(test_file['type'])
# test_file = test_file.drop(['id'], axis =1) #
# test_file = test_file.drop(['citric acid','sulphates'], axis =1) #,<-------- 상관관계
# test_file = scaler.transform(test_file)

# num = test_file.shape[0]
# test_file = test_file.reshape(num,2,5,1) 
# # test_file = test_file.to_numpy()

# # ############### 제출용.
# result = model.predict(test_file)
# # print(result[:5])

# result_recover = np.argmax(result, axis = 1).reshape(-1,1) + 4
# # print(result_recover[:5])
# print(np.unique(result_recover)) # np.unique()
# submission['quality'] = result_recover

# acc_list = hist.history['accuracy']
# acc = opt + "_acc_"+str(acc_list[-patience_num]).replace(".", "_")
# print(acc)
# # acc= str(loss[1]).replace(".", "_")
# model.save(f"./_save/_dacon_save_model_{acc}.h5")
# submission.to_csv(f"./_save/{opt}_dacon_{acc}.csv", index = False)
# '''

# '''
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

import numpy as np
import pandas as pd  
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(10,10))
# sns.heatmap(data= train.corr(), square=True, annot=True, cbar=True)
# plt.show()

x = train.drop(['id','quality','citric acid','sulphates'], axis=1) 
print(x.columns)

test_file = test_file.drop(['id','citric acid','sulphates'], axis=1)

#print(x.shape)    # (3231, 10)

y = train['quality']
#print(y.shape)  # (3231,)

le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])


#label = x['type']
#le.fit(label)
#x['type'] = le.transform(label)
#print(x)


#로그변환
#y = np.log1p(y)   #log는 0이 되면 노노.. 그래서 1더해줘야함! log1p는 자동으로 +1해줌 
#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)   #One Hot Encoding
#print(y)
#print(y.shape)


from pandas import get_dummies 
y = get_dummies(y)
#print(y.shape)  # (3231, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66)



scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train) 

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2)모델

input1 = Input(shape=(10,))
dense1 = Dense(90, activation = 'relu')(input1)
dense2 = Dense(110, activation = 'relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(130, activation = 'relu')(dense2)
drop2 = Dropout(0.5)(dense3)
dense4 = Dense(150)(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(110)(drop3)
drop4 = Dropout(0.5)(dense5)
dense6 = Dense(90, activation = 'relu')(drop4)
output1 = Dense(5, activation='softmax')(drop4)
model = Model(inputs=input1, outputs=output1)


#3)컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'Nadam', metrics=['accuracy'])

es = EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min', verbose=1, restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.25, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# rmse = RMSE(y_test, y_pred)
# print("RMSE: ", rmse)  


#loss:  [1.047375202178955, 0.5486862659454346]

# loss:  [1.00103759765625, 0.5687789916992188]

# loss:  [0.9904183149337769, 0.5687789916992188]
###================================================== 제출용


result = model.predict(test_file)
#print(result)
result_int = np.argmax(result, axis =1).reshape(-1,1) + 4 # 결과를 열로 뽑겠따!
submit_file['quality'] = result_int

# argmax: 원핫인코딩된 데이터를 결과데이터에 넣을때 다시 숫자로, 되돌려 주는 편리한 기능을 제공해주는 함수 / 확률을 다시 change
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv( path +f"result/accuracy_{acc}.csv", index=False)  # 디폴트: 기본으로 index가 생성됨 / if index= false하면 인덱스 생성x

