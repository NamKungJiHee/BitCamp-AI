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

#D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/rps/rps/', 
    target_size = (150,150), 
    batch_size=5,
    class_mode = 'categorical',seed=66, color_mode='grayscale',
    save_to_dir='../_temp/', shuffle= True)   ############# save_to_dir: 이는 디렉토리를 선택적으로 지정해서 생성된 증강 사진을 저장할 수 있도록 합니다.


xy_test = test_datagen.flow_from_directory(
    '../_data/image/rps/rps/',
    target_size = (150,150),
    batch_size = 5, 
    class_mode = 'categorical') 


print(xy_train)   #x와 y가 함께 있음
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E743074F40>

# # from sklearn.datasets import load_boston
# # datasets = load_boston()
# # print(datasets)

#print(xy_train[31]) # 마지막 배치 출력 (배치 사이즈가 곧 y 갯수)  train에는 총 160개있는데 배치 5로 나눠서 0~31까지 있음
print(xy_train[0][0])  # [0]번째 배치의 [0] x값
print(xy_train[0][1])  # [0]번째 배치의 [1] y값
# # print(xy_train[0][2])  # IndexError: tuple index out of range
#print(xy_train[0][0].shape, xy_train[0][1].shape)  # (5, 150, 150, 1) (5, 2)

# print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))  # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

