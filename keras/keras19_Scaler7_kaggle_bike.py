<<<<<<< HEAD
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd  
from sklearn.metrics import r2_score, mean_squared_error


def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test,y_pred)) 


#1)데이터
path = "./_data/bike/"      #.지금 현재 작업 공간   / ..이전


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

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움
test_file = scaler.transform(test_file)

#2)모델
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 8))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

#model.summary()
#3)컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

rmse = RMSE(y_test, y_pred)
print("RMSE: ", rmse)  


"""
# 결과       loss/accuracy
# 그냥
loss:  1.4901468753814697
r2:  0.23954803500258293
RMSE:  1.2207157865965643

# MinMAX
loss:  1.4562679529190063
r2:  0.2568371273335496
RMSE:  1.2067593437968513

#Standard
loss:  1.4643399715423584
r2:  0.2527177415350852
RMSE:  1.2100992829564106

# robust 
loss:  1.4604939222335815
r2:  0.25468054219703207
RMSE:  1.2085090234834857

# maxabs
loss:  1.4572277069091797
r2:  0.256347333321136
RMSE:  1.207156945834871

== minmax scaler로 돌렸을 때 최소의 loss가 나오긴하지만 다 비슷비슷한 결과가 나온다!

##################################### relu 적용 후 
# 그냥
loss:  1.4309674501419067
r2:  0.2697486385134137
RMSE:  1.1962304563330017

# MinMAX
loss:  1.396662950515747
r2:  0.2872549201180912
RMSE:  1.1818048877229756

#Standard
loss:  1.3675107955932617
r2:  0.3021318347791482
RMSE:  1.1694061177909598

# robust 
loss:  1.3669248819351196
r2:  0.3024307185526741
RMSE:  1.1691556736802102

# maxabs
loss:  1.3741101026535034
r2:  0.29876401608834646
RMSE:  1.17222442067533

== relu함수를 쓰고 나서 loss의 값이 조금 더 떨어졌따! 

######################summary

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                90
_________________________________________________________________
dense_1 (Dense)              (None, 20)                220
_________________________________________________________________
dense_2 (Dense)              (None, 30)                630
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 31
=================================================================
Total params: 971
Trainable params: 971
Non-trainable params: 0
"""

###================================================== 제출용
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

=======
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd  
from sklearn.metrics import r2_score, mean_squared_error


def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test,y_pred)) 


#1)데이터
path = "./_data/bike/"      #.지금 현재 작업 공간   / ..이전


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

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train에 맞는 비율로 들어가있움
test_file = scaler.transform(test_file)

#2)모델
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 8))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

#model.summary()
#3)컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

rmse = RMSE(y_test, y_pred)
print("RMSE: ", rmse)  


"""
# 결과       loss/accuracy
# 그냥
loss:  1.4901468753814697
r2:  0.23954803500258293
RMSE:  1.2207157865965643

# MinMAX
loss:  1.4562679529190063
r2:  0.2568371273335496
RMSE:  1.2067593437968513

#Standard
loss:  1.4643399715423584
r2:  0.2527177415350852
RMSE:  1.2100992829564106

# robust 
loss:  1.4604939222335815
r2:  0.25468054219703207
RMSE:  1.2085090234834857

# maxabs
loss:  1.4572277069091797
r2:  0.256347333321136
RMSE:  1.207156945834871

== minmax scaler로 돌렸을 때 최소의 loss가 나오긴하지만 다 비슷비슷한 결과가 나온다!

##################################### relu 적용 후 
# 그냥
loss:  1.4309674501419067
r2:  0.2697486385134137
RMSE:  1.1962304563330017

# MinMAX
loss:  1.396662950515747
r2:  0.2872549201180912
RMSE:  1.1818048877229756

#Standard
loss:  1.3675107955932617
r2:  0.3021318347791482
RMSE:  1.1694061177909598

# robust 
loss:  1.3669248819351196
r2:  0.3024307185526741
RMSE:  1.1691556736802102

# maxabs
loss:  1.3741101026535034
r2:  0.29876401608834646
RMSE:  1.17222442067533

== relu함수를 쓰고 나서 loss의 값이 조금 더 떨어졌따! 

######################summary

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                90
_________________________________________________________________
dense_1 (Dense)              (None, 20)                220
_________________________________________________________________
dense_2 (Dense)              (None, 30)                630
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 31
=================================================================
Total params: 971
Trainable params: 971
Non-trainable params: 0
"""

###================================================== 제출용
results = model.predict(test_file)

submit_file['count'] = results

print(submit_file[:10])

>>>>>>> b6273d91f0d2a8bda64398dfce3bbe5e3e083b07
submit_file.to_csv( path + "submitfile.csv", index=False)