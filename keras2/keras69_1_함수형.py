from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
import time
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

## [실습] ##

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)  
#print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32,32,3)         
x_test = x_test.reshape(10000,32,32,3)           
#print(x_train.shape)     # (50000, 32, 32, 3)
     
#print(np.unique(y_train, return_counts=True))   #  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
#print(y_train.shape)  
y_test = to_categorical(y_test)
#print(y_test.shape)

scaler= StandardScaler()              
x_train= x_train.reshape(50000,-1)   
x_test = x_test.reshape(10000,-1)    
                                   
x_train=scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)

#2. 모델링
input = Input(shape = (32, 32, 3))
vgg16 = VGG16(include_top= False)(input)
hidden = Flatten()(vgg16)
hidden1 = Dense(100)(hidden)
hidden2 = Dense(80, activation = 'relu')(hidden1)
hidden3 = Dense(60)(hidden2)
hidden4 = Dense(40)(hidden3)
output1 = Dense(100, activation = 'softmax')(hidden4)

model = Model(inputs = input, outputs = output1)

#model.summary()

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

''' 
learning_rate:  0.0001
loss:  1.8842
accuracy:  0.531
걸린시간:  288.683
'''