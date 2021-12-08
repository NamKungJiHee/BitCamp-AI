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
dense1 = Dense(30)(input1)
dense2 = Dense(50, activation = 'relu')(dense1)
dense3 = Dense(70, activation = 'relu')(dense2)
drop1 = Dropout(0.5)(dense3)
dense4 = Dense(10)(drop1)
drop2 = Dropout(0.2)(dense4)
dense6 = Dense(5)(drop2)
output1 = Dense(5, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)


#3)컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.25, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)




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
