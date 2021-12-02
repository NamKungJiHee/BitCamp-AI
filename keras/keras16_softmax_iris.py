<<<<<<< HEAD
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_iris()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))   # [0 1 2] = 다중분류      #이걸 해봐야 이진분류인지 다중분류인지 알 수 있움!!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)



#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 4))     
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
loss:  0.10424169898033142
accuracy:  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[9.3144458e-04 9.9653012e-01 2.5384326e-03]
 [9.0801484e-05 9.7805917e-01 2.1850070e-02]
 [1.5423632e-04 9.6235907e-01 3.7486747e-02]
 [9.9933177e-01 6.6815189e-04 3.1080190e-17]
 [3.1696740e-04 9.9589008e-01 3.7929991e-03]
 [2.1666931e-03 9.9604934e-01 1.7839626e-03]
 [9.9946147e-01 5.3849758e-04 8.3059813e-17]]
=======
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_iris()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))   # [0 1 2] = 다중분류

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)



#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 4))     
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
loss:  0.10424169898033142
accuracy:  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[9.3144458e-04 9.9653012e-01 2.5384326e-03]
 [9.0801484e-05 9.7805917e-01 2.1850070e-02]
 [1.5423632e-04 9.6235907e-01 3.7486747e-02]
 [9.9933177e-01 6.6815189e-04 3.1080190e-17]
 [3.1696740e-04 9.9589008e-01 3.7929991e-03]
 [2.1666931e-03 9.9604934e-01 1.7839626e-03]
 [9.9946147e-01 5.3849758e-04 8.3059813e-17]]
=======
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_iris()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))   # [0 1 2] = 다중분류

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)



#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 4))     
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
loss:  0.10424169898033142
accuracy:  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[9.3144458e-04 9.9653012e-01 2.5384326e-03]
 [9.0801484e-05 9.7805917e-01 2.1850070e-02]
 [1.5423632e-04 9.6235907e-01 3.7486747e-02]
 [9.9933177e-01 6.6815189e-04 3.1080190e-17]
 [3.1696740e-04 9.9589008e-01 3.7929991e-03]
 [2.1666931e-03 9.9604934e-01 1.7839626e-03]
 [9.9946147e-01 5.3849758e-04 8.3059813e-17]]
>>>>>>> b6273d91f0d2a8bda64398dfce3bbe5e3e083b07
'''