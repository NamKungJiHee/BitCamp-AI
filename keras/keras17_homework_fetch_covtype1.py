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

model.fit(x_train, y_train, epochs = 100, validation_split= 0.2, callbacks=[es])   # 데이터가 크기 때문에 batch_size를 크게 해줌!

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
results= model.predict(x_test[:11])
print(y_test[:11])
print('results: ', results)


'''
Q) batch_size의 디폴트는 몇??  31.9978... = 32
batch_size를 1로 했을 때 1epoch 당 371847
batch_size를 완전히 지우고 돌렸을 때 1epoch당 11621
371847/11621 = 31.9978... = 32
'''