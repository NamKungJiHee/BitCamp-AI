<<<<<<< HEAD
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_wine()
#print(datasets.DESCR) # (178, 13) (178,)  <----y=1이니까..   ----> (178, 3)으로 바꿔야 함   [0,1,2]니까 output이 3으로 바껴야 한다..
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))   # [0 1 2]  y를 써주는 이유: y가 출력값이므로 이진인지 다중인지 알 수 있움!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (178, 3)   원핫인코딩으로 3열로 바꿔줌

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)     ####고냥 확인하려구!
#print(x_test.shape, y_test.shape)    # (36, 13) (36, 3)


#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 13))     
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(3, activation = 'softmax'))

#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측


loss=model.evaluate(x_test, y_test)  
print('loss: ', loss[0]) # 원래 [loss, accuracy] 이렇게 나오므로...
print('accuracy: ', loss[1])

results=model.predict(x_test[:7])
print(y_test[:7])  
print(results)

'''
loss:  0.22500453889369965
accuracy:  0.9166666865348816     ex) 개, 고양이, 사람
[[0. 0. 1.]       사람
 [0. 1. 0.]       고양이     
 [0. 1. 0.]
 [1. 0. 0.]       개
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[3.34817654e-04 1.46158114e-01 8.53507042e-01]    이 셋중에 세번째거가 젤 높으므로 [0,0,1]이 됨
 [1.55019432e-01 8.05277109e-01 3.97034474e-02]
 [4.47165294e-05 9.99509811e-01 4.45489102e-04]
 [9.96494949e-01 1.47262635e-03 2.03236006e-03]
 [2.90702184e-04 9.97155547e-01 2.55370536e-03]
 [4.67943282e-05 9.99127686e-01 8.25554773e-04]
 [1.18345104e-01 5.99239692e-02 8.21730912e-01]]
=======
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_wine()
#print(datasets.DESCR) # (178, 13) (178,)  <----y=1이니까..   ----> (178, 3)으로 바꿔야 함   [0,1,2]니까 output이 3으로 바껴야 한다..
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))   # [0 1 2] 

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (178, 3)   원핫인코딩으로 3열로 바꿔줌

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)     ####고냥 확인하려구!
print(x_test.shape, y_test.shape)    # (36, 13) (36, 3)


#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 13))     
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(3, activation = 'softmax'))

#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측


loss=model.evaluate(x_test, y_test)  
print('loss: ', loss[0]) # 원래 [loss, accuracy] 이렇게 나오므로...
print('accuracy: ', loss[1])

results=model.predict(x_test[:7])
print(y_test[:7])  
print(results)

'''
'''
loss:  0.22500453889369965
accuracy:  0.9166666865348816     ex) 개, 고양이, 사람
[[0. 0. 1.]       사람
 [0. 1. 0.]       고양이     
 [0. 1. 0.]
 [1. 0. 0.]       개
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[3.34817654e-04 1.46158114e-01 8.53507042e-01]    이 셋중에 세번째거가 젤 높으므로 [0,0,1]이 됨
 [1.55019432e-01 8.05277109e-01 3.97034474e-02]
 [4.47165294e-05 9.99509811e-01 4.45489102e-04]
 [9.96494949e-01 1.47262635e-03 2.03236006e-03]
 [2.90702184e-04 9.97155547e-01 2.55370536e-03]
 [4.67943282e-05 9.99127686e-01 8.25554773e-04]
 [1.18345104e-01 5.99239692e-02 8.21730912e-01]]

=======
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_wine()
#print(datasets.DESCR) # (178, 13) (178,)  <----y=1이니까..   ----> (178, 3)으로 바꿔야 함   [0,1,2]니까 output이 3으로 바껴야 한다..
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))   # [0 1 2] 

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (178, 3)   원핫인코딩으로 3열로 바꿔줌

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)     ####고냥 확인하려구!
print(x_test.shape, y_test.shape)    # (36, 13) (36, 3)


#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 13))     
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(3, activation = 'softmax'))

#3) 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측


loss=model.evaluate(x_test, y_test)  
print('loss: ', loss[0]) # 원래 [loss, accuracy] 이렇게 나오므로...
print('accuracy: ', loss[1])

results=model.predict(x_test[:7])
print(y_test[:7])  
print(results)

'''
loss:  0.22500453889369965
accuracy:  0.9166666865348816     ex) 개, 고양이, 사람
[[0. 0. 1.]       사람
 [0. 1. 0.]       고양이     
 [0. 1. 0.]
 [1. 0. 0.]       개
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[3.34817654e-04 1.46158114e-01 8.53507042e-01]    이 셋중에 세번째거가 젤 높으므로 [0,0,1]이 됨
 [1.55019432e-01 8.05277109e-01 3.97034474e-02]
 [4.47165294e-05 9.99509811e-01 4.45489102e-04]
 [9.96494949e-01 1.47262635e-03 2.03236006e-03]
 [2.90702184e-04 9.97155547e-01 2.55370536e-03]
 [4.67943282e-05 9.99127686e-01 8.25554773e-04]
 [1.18345104e-01 5.99239692e-02 8.21730912e-01]]
>>>>>>> b6273d91f0d2a8bda64398dfce3bbe5e3e083b07
'''