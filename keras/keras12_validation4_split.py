
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

#train_test_split로 나누시오 (10:3:3)


'''
x_train = x[:11]
y_train = y[:11]
x_test = x[11:14]
y_test = y[11:14]
x_val = x[14:17]
y_val = y[14:17]
'''

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8125, shuffle=True, random_state=66)
#13개 3개


#x_train, x_val, y_train, y_val = train_test_split(x_test,y_test, train_size=0.5, random_state=66)



print(x_train)
print(x_test)
#print(x_val)

#2. 모델구성
model = Sequential()
model.add(Dense(21,input_dim=1))
model.add(Dense(17))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련                  
#train_test_split로 나누시오 (10:3:3)              
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)       #validation_data=(x_val, y_val))   #loss보다 val_loss를 더 신뢰!!! 

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([18])
print("18의 예측값: ", y_predict)



#loss:  3.07901843127692e-13
#18의 예측값:  [[18.]]