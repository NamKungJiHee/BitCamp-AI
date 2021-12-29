from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0]

# print(x_augumented)
# print(x_augumented.shape)  # (40000, 28, 28, 1)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augmented))
#print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)





