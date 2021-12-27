import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.layers.core import Dropout


#1. 데이터
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

#D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/', 
    target_size = (150,150), 
    batch_size=5,
    class_mode = 'binary',
    shuffle= True) 
#target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로 
#"categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 160 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150,150),
    batch_size = 5,
    class_mode = 'binary'
    ) 
#####Found 120 images belonging to 2 classes.#####

print(xy_train)   #x와 y가 함께 있음
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E743074F40>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

#print(xy_train[31]) # 마지막 배치 출력 (배치 사이즈가 곧 y 갯수)  train에는 총 160개있는데 배치 5로 나눠서 0~31까지 있음
# print(xy_train[0][0])  # [0]번째 배치의 [0] x값
# print(xy_train[0][1])  # [0]번째 배치의 [1] y값
# print(xy_train[0][2])  # IndexError: tuple index out of range
print(xy_train[0][0].shape, xy_train[0][1].shape)  # x = (5, 150, 150, 3) # 배치, 가로, 세로, 채널(컬러) &  y = (5,)  ==> 이진분류이므로 (5,2)로 바꿀 수 있음

'''
print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
'''

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (150,150,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
#model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs= 200, steps_per_epoch= 32, validation_data= xy_test,
                    validation_steps= 4, callbacks = [es])  
                       # batch_size는 이미 위에서 5로 주었기 때문에 명시x 
                       # steps_per_epoch=배치/전체 데이터  ex) 160/5=32
                       # validation_steps=한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정/steps_per_epoch이 특정된 경우에만 유의미
                       

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 그래프 그리기
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])

''' 
loss:  0.623172402381897
val_loss:  0.6858189105987549
acc:  0.668749988079071
val_acc:  0.5
'''