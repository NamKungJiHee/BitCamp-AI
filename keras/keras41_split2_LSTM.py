import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM

a = np.array(range(1,101))
x_pred = np.array(range(96,106))


#size = 5  # x 4개, y 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):     # 반복횟수
        subset = dataset[i : (i +size)]        # slicing # RNN
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, 5)       
# print(bbb)
# print(bbb.shape)   # (6, 5) 
      
x = bbb[:, :4]  # 행, 열 
y = bbb[:, 4]   # 행, 열
# print(x,y) 
#print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(96,4,1)

ab = split_x(x_pred,4)
x_pred = ab.reshape(7,4,1)
print(x_pred.shape)  #(7,4,1)

#모델구성 LSTM형식으로!
model = Sequential()
model.add(LSTM(80, input_shape = (4,1))) 
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer= 'adam') 
model.fit(x,y,epochs=100)  

#4. 평가, 예측
model.evaluate(x,y)

x_predict = model.predict(x_pred)
print(x_predict)



