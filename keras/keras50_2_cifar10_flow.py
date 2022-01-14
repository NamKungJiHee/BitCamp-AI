# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 _temp에 넣고 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')

augment_size = 50000

# print(x_train[0].shape) # (32, 32, 3)
# print(x_train[0].reshape(32*32*3).shape) # (3072,)
#print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1) 

randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다. 

# print(randidx) # [28513  5111 46874 ... 14341 36003 12194]
# print(np.min(randidx), np.max(randidx)) # 0 49999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augmented.shape) # (50000, 32, 32, 3)
# print(y_augmented.shape) # (50000, 1)

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0] #save_to_dir='../_temp/').next()[0]  # 증폭

#print(x_augmented)
# print(x_augmented.shape)  # (50000, 32, 32, 3)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (100000, 32, 32, 3) (100000, 1)

#print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

#2. 모델
model = Sequential()
model.add(Conv2D(128,(2,2), input_shape= (32,32,3)))
model.add(Conv2D(64,(2,2)))
model.add(Conv2D(32,(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.25, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

''' 
loss:  [2.233361005783081, 0.33180001378059387]
'''