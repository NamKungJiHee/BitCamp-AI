import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.layers.core import Dropout


#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255) 
    # horizontal_flip = True,
    # vertical_flip = True,
    # width_shift_range= 0.1,
    # height_shift_range= 0.1,
    # rotation_range= 5,
    # zoom_range = 1.2,
    # shear_range = 0.7,
    # fill_mode= 'nearest')   # 1./255(scaler역할)  , zoom_range(확대)
 
test_datagen = ImageDataGenerator(rescale=1./255) # 평가할 때는 원래의 데이터(이미지)로 해야하므로 많은 조건을 줄 필요가 없음.(증폭할 필요 없음)

#D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/', 
    target_size = (150,150), 
    batch_size=200,     # 일부러 배치를 크게 줌
    class_mode = 'binary',
    shuffle= True) 
#target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로 
#"categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 160 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150,150),
    batch_size = 200,
    class_mode = 'binary'
    ) 
#####Found 120 images belonging to 2 classes.#####

print(xy_train[0][0].shape, xy_train[0][1].shape) # (160, 150, 150, 3)   (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)  # (120, 150, 150, 3)    (120,)

np.save('./_save_npy/keras47_5_train_x.npy', arr = xy_train[0][0])   # arr = 변수명
np.save('./_save_npy/keras47_5_train_y.npy', arr = xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy', arr = xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy', arr = xy_test[0][1])

