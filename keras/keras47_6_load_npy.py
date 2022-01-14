import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

# np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])   # arr = 변수명
# np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')
# print(x_train)
# print(x_train.shape)  # (160, 150, 150, 3)
# print(y_train.shape)  # (160,)
# print(x_test.shape)  # (120, 150, 150, 3)
# print(y_test.shape)  # (120,)

#2. 모델

model = Sequential()
model.add(Conv2D(35, (2,2), input_shape= (150,150,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

model.fit(x_train,y_train, epochs = 100, validation_split=0.2, callbacks=[es], batch_size = 50)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

# y_predict = model.predict(x_test)
# print('예측값: ', y_predict)

''' 
loss: [0.03845858573913574, 0.9916666746139526]

'''

