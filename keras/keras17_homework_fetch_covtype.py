import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (581012, 54) (581012,)  ----> (다중분류) 
print(np.unique(y))    # [1 2 3 4 5 6 7]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape) # (581012, 8) 원핫인코딩


#2) 모델링
model = Sequential()
model.add(Dense(10, activation= 'linear', input_dim = 54))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(8, activation = 'softmax'))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#3) 캄파일, 훈련

model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs = 100, batch_size=500, validation_split= 0.2, callbacks=[es])   # 데이터가 크기 때문에 batch_size를 크게 해줌!

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
results= model.predict(x_test[:11])
print(y_test[:11])
print('results: ', results)

'''
744/744 [==============================] - 1s 821us/step - loss: 0.6520 - accuracy: 0.7172 - val_loss: 0.6491 - val_accuracy: 0.7199
3632/3632 [==============================] - 1s 386us/step - loss: 0.6499 - accuracy: 0.7191
loss:  [0.6499490737915039, 0.7190605998039246]
[[0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]]
results:  [[6.7524980e-16 6.6217667e-01 2.9989788e-01 6.9926735e-08 1.5794003e-12
  1.1370080e-03 7.8764913e-08 3.6788240e-02]
 [8.0859798e-12 3.7245445e-02 9.6060568e-01 5.7736212e-05 7.9180609e-06
  9.3780167e-04 1.1202955e-03 2.5135054e-05]
 [1.0307360e-12 7.5254965e-01 2.2166377e-01 1.7359546e-06 7.7425732e-11
  9.3061122e-04 1.5524378e-05 2.4838697e-02]
 [5.4601358e-12 5.4371752e-02 9.4169724e-01 9.2514907e-05 2.2003461e-04
  2.8448498e-03 5.0500710e-04 2.6850207e-04]
 [1.2903365e-11 4.0505141e-01 5.7441771e-01 4.8673170e-05 2.7603772e-10
  4.8135207e-03 3.2808926e-05 1.5635928e-02]
 [4.4030704e-18 2.3243770e-01 7.3034692e-01 1.5331620e-05 7.3920484e-14
  3.7117127e-02 2.6782258e-05 5.6070308e-05]
 [3.5240129e-12 1.5809011e-01 8.3906478e-01 1.5715137e-05 5.2283320e-07
  2.8229642e-04 3.2707085e-05 2.5138438e-03]
 [1.5644233e-10 3.0658476e-03 2.0516174e-01 6.1376047e-01 5.5428065e-04
  8.3300523e-02 9.4152734e-02 4.4771132e-06]
 [1.6619260e-10 1.5561900e-01 8.3111429e-01 8.2351553e-04 9.6170697e-07
  9.8817525e-03 1.4583268e-03 1.1021148e-03]
 [8.1858659e-10 6.2847015e-08 6.5266569e-05 7.3870158e-01 1.7586060e-02
  2.1190772e-05 2.4362582e-01 3.6224686e-12]
 [2.0891075e-18 7.9661155e-01 2.0217665e-01 8.4732221e-09 1.0423681e-14
  7.6652352e-05 1.5130217e-07 1.1349684e-03]]
'''
