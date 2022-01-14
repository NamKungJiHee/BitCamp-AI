<<<<<<< HEAD
from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다.  0~ 60000만개 중에 40000만개의 값을 랜덤으로 뽑겠다.(중복 포함 x) = 증폭
#print(x_train.shape[0]) # 60000 
#print(randidx) # [32487  8152 30682 ... 47171 47203 53853]
#print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augumented.shape) # (40000, 28, 28)
# print(y_augumented.shape) # (40000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

xy_train = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= 32,#augment_size,
                                  shuffle= False) #.next()       # flow를 지나고 보니 튜플 형식으로 묶여서 나온다.

# print(xy_train) 
# print(xy_train[0].shape, xy_train[1].shape)

#2. 모델
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape= (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy 일케 써주면 y의 원핫인코딩을 안해도 된다.

#print(len(xy_train)) #1250   (40000/32)

model.fit_generator(xy_train, epochs = 10, steps_per_epoch = len(xy_train)) # (40000/32)

#4. 평가, 예측

# loss = model.evaluate_generator(x_test, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
# # https://keras.io/ko/models/sequential/

# pred = model.predict_generator(x_test)


=======
from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다.  0~ 60000만개 중에 40000만개의 값을 랜덤으로 뽑겠다.(중복 포함 x) = 증폭
#print(x_train.shape[0]) # 60000 
#print(randidx) # [32487  8152 30682 ... 47171 47203 53853]
#print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augumented.shape) # (40000, 28, 28)
# print(y_augumented.shape) # (40000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

xy_train = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= 32,#augment_size,
                                  shuffle= False) #.next()       # flow를 지나고 보니 튜플 형식으로 묶여서 나온다.

# print(xy_train) 
# print(xy_train[0].shape, xy_train[1].shape)

#2. 모델
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape= (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy 일케 써주면 y의 원핫인코딩을 안해도 된다.

#print(len(xy_train)) #1250   (40000/32)

model.fit_generator(xy_train, epochs = 10, steps_per_epoch = len(xy_train)) # (40000/32)

#4. 평가, 예측

# model.evaluate_generator(x_test, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
# # https://keras.io/ko/models/sequential/

# pred = model.predict_generator(x_test)


>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
