import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional,Conv1D,Flatten

#1. 데이터
x = np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])   
y = np.array([4,5,6,7])

print(x.shape,y.shape)  # (4, 3) (4,)

# input_shape = (행, 열, 몇개씩 자르는지!!)  = 3차원(내용과 순서 안바뀜 = reshape)
x = x.reshape(4, 3, 1) # 4행 3열 1개씩 자르겠다 ([[1],[2],[3]],...)  ---> 여기서 2차원을 3차원으로 바꿔줌

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(80, input_shape = (3,1),return_sequences=True)) 
#model.add(Bidirectional(SimpleRNN(10),input_shape=(3,1)))  # SimpleRNN을 양방향으로 돌리겠다.
model.add(Conv1D(10, 2, input_shape= (3,1)))  # 2: 2가 kernel size.. 2씩 묶어주겠다. & step / 10: filter
model.add(Dense(60, activation = 'relu')) 
model.add(Flatten())
model.add(Dense(1))

model.summary()

""" 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 2, 10)             30
_________________________________________________________________
dense (Dense)                (None, 2, 60)             660
_________________________________________________________________
flatten (Flatten)            (None, 120)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 121
=================================================================
Total params: 811
Trainable params: 811
Non-trainable params: 0
"""
#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer= 'adam') 
# model.fit(x,y,epochs=100)

# # #4. 평가, 예측
# model.evaluate(x,y)
# result = model.predict([[[5],[6],[7]]])
# print(result)

