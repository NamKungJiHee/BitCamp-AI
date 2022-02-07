from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

#1) 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)  
#print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)  


#print(np.unique(y_train, return_counts=True)) #  다중분류

y_train = to_categorical(y_train)   
#print(y_train.shape)   #(60000, 10)
y_test = to_categorical(y_test)
#print(y_test.shape)

# n = x_train.shape[0]
# x_train = x_train.reshape(n,-1)/255.       

# m = x_test.shape[0]
# x_test = x_test.reshape(m,-1)/255.

# print(x_train.shape, x_test.shape)

#2) 모델링

model = Sequential()
model.add(LSTM(80, input_shape = (28,28))) 
model.add(Dense(45))
model.add(Dense(10, activation ='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss = 'categorical_crossentropy', optimizer =  optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
hist = model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es, reduce_lr], batch_size = 50)
end = time.time()

#4 평가예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))
################################# 시각화 #################################
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1.
plt.subplot(2,1,1)  # 2행 1열 중에 첫번째꺼
plt.plot(hist.history['loss'], marker = ',', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = ',', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss') # y값
plt.xlabel('epochs') # x값
plt.legend(loc = 'upper right') # 그래프 위에 생성되는 것!

#2.
plt.subplot(2,1,2)  # 2행 1열 중에 두번째꺼
plt.plot(hist.history['accuracy'], marker = ',', c = 'red', label = 'loss')
plt.plot(hist.history['val_accuracy'], marker = ',', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epochs') 
plt.legend(['accuracy', 'val_accuracy'])

plt.show()