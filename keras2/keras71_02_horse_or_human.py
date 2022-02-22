from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121, VGG19
import time
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')

path = '../_data/image/horseorhuman/horse-or-human/horse-or-human'

x_train = np.load('../_data/_save_npy/keras48_2_1_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_2_1_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_2_1_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_2_1_test_y.npy')

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # (600, 150, 150, 3) (600, 2) (205, 150, 150, 3) (205, 2)

x_train = preprocess_input(x_train)  
x_test = preprocess_input(x_test)
print("===================preprocess_input(x)=======================")
print(x_train.shape, x_test.shape)

vgg19 = VGG19(weights = 'imagenet', include_top= False, input_shape= (150, 150, 3))

#densenet121.trainable = False  # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg19)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(2, activation = 'sigmoid'))
    
#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

''' 
learning_rate:  0.0001
loss:  1.7302
accuracy:  0.7561
걸린시간:  87.5987
=================preprocess_input=================
learning_rate:  0.0001
loss:  0.6933
accuracy:  0.4878
걸린시간:  51.1251
'''