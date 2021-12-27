# http://www.kaggle.com/c/dogs-vs-cats/data

from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode= 'nearest')   # 1./255(scaler역할)  , zoom_range(확대)
 
test_datagen = ImageDataGenerator(rescale=1./255) # 평가할 때는 원래의 데이터(이미지)로 해야하므로 많은 조건을 줄 필요가 없음.(증폭할 필요 없음)

#D:\_data\image\catdog\cat_dog\

xy_train = train_datagen.flow_from_directory(
    '../_data/image/catdog/cat_dog', 
    target_size = (100,100), 
    batch_size=23,
    class_mode = 'binary',
    shuffle= True
) 
#target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로 
#"categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 10028 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '../_data/image/catdog/cat_dog',
    target_size = (100,100),
    batch_size = 23,
    class_mode = 'binary') 
#####Found  10028 images belonging to 2 classes.#####

print(xy_train)   #x와 y가 함께 있음
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E743074F40>

print(xy_train[0][0].shape, xy_train[0][1].shape)  # x = (23, 100, 100, 3) (23,) 
#print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))  # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (100,100,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs= 200, steps_per_epoch= 436, validation_data= xy_test,
                    validation_steps= 4, callbacks = [es])  
                       

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])

''' 
loss:  0.5317307710647583
val_loss:  0.4497367739677429
acc:  0.793749988079071
val_acc:  0.8500000238418579
'''
