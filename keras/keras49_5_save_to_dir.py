<<<<<<< HEAD
from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

########### print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)############
#### augment ####

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

augment_size = 10
randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다.  
#print(x_train.shape[0]) # 60000 
#print(randidx) # [32487  8152 30682 ... 47171 47203 53853]
#print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
print(x_augmented.shape) # (400000, 28, 28)
print(y_augmented.shape) # (400000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

import time

start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False, save_to_dir='../_temp/').next()[0]

end_time = time.time()-start_time

print("걸린시간: ", round(end_time,3), "초") 


=======
from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

########### print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)############
#### augment ####

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

augment_size = 10
randidx = np.random.randint(x_train.shape[0], size = augment_size) # 랜덤한 정수값 생성하겠다.  
#print(x_train.shape[0]) # 60000 
#print(randidx) # [32487  8152 30682 ... 47171 47203 53853]
#print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
print(x_augmented.shape) # (400000, 28, 28)
print(y_augmented.shape) # (400000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

import time

start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False, save_to_dir='../_temp/').next()[0]

end_time = time.time()-start_time

print("걸린시간: ", round(end_time,3), "초") 


>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
