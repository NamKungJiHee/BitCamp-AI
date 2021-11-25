from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터
x = np.array(range(100))
y = np.array(range(1,101))

#train : test = 7:3

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size=0.7, shuffle=True, random_state=66)  #random_state: 일정한 값이 나오게 해줌

print(x_test) #[ 8 93  4  5 52 41  0 73 88 68]
print(y_test) 

model = Sequential()
model.add(Dense(3, input_dim=1))  
model.add(Dense(30)) 
model.add(Dense(7))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train ,y_train , epochs=100, batch_size=1)


#04. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict([100]) 
print('101의 예측값: ', y_predict)

#loss:  1.336222937853293e-10
#101의 예측값:  [[101.00004]]


#다층 퍼셉트론(multi-layer perceptron, MLP)는 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태
#퍼셉트론: 퍼셉트론은 다수의 신호를 입력받아 하나의 신호를 출력하는 알고리즘

#shift + delete = 한라인 삭제
#ctrl + / = 주석처리