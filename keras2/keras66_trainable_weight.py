import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim = 1))   # 한 layer당 weight와 bias가 있는데 이게 3세트가 있으므로 아래에 숫자가 6이 나오는 것!!
model.add(Dense(2))
model.add(Dense(1))

model.summary()    # 1 * 3 + 3 = 6

print(model.weights)
print("====================================================")
print(model.trainable_weights)
print("====================================================")

print(len(model.weights))  # 6
print(len(model.trainable_weights))  # 6

model.trainable = False   # 가중치 갱신을 하지 않겠다. = loss값이 그대로다. 가중치 갱신x

print(len(model.weights))  # 6
print(len(model.trainable_weights))  # 0

model.summary()

model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, batch_size = 1, epochs = 100)