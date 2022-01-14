
# http://www.kaggle.com/c/dogs-vs-cats/data

from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip = True,  # 상하반전
    vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode= 'nearest')   # 1./255(scaler역할)  , zoom_range(확대)
 
test_datagen = ImageDataGenerator(rescale=1./255) # 평가할 때는 원래의 데이터(이미지)로 해야하므로 많은 조건을 줄 필요가 없음.(증폭할 필요 없음)

xy_train = train_datagen.flow_from_directory(
    '../_data/image/catdog/cat_dog/training_set/training_set/', 
    target_size = (100,100), 
    batch_size=23,
    class_mode = 'categorical',
    shuffle= True
) 
#target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로 
#"categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 10028 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '../_data/image/catdog/cat_dog/test_set/test_set/',
    target_size = (100,100),
    batch_size = 23,
    class_mode = 'categorical') 
#####Found  10028 images belonging to 2 classes.#####

#print(xy_train)   #x와 y가 함께 있음
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E743074F40>

#print(xy_train[0][0].shape, xy_train[0][1].shape)  # x = (23, 100, 100, 3)  y = (23,) 
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
model.add(Conv2D(128, (2,2), input_shape= (100,100,3)))
model.add(Conv2D(64, (2,2), padding='same'))  
model.add(Conv2D(16, (2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs= 200, steps_per_epoch= 436, validation_data= xy_test,
                    validation_steps= 4, callbacks = [es])  
                       
model.save("./_save/keras48_1_save_weights1111.h5")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])

''' 
loss:  0.7859594225883484
val_loss:  0.6968246698379517
acc:  0.5309181809425354
val_acc:  0.489130437374115
'''

import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/catdog/predict/cat.4003.jpg'
model_path = './_save/keras48_1_save_weights1111.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(100,100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    cat = pred[0][0]*100
    dog = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
        
''' 
loss:  3.9810919761657715
val_loss:  0.7714969515800476
acc:  0.5166770815849304
val_acc:  0.6086956262588501
당신은 50.51 % 확률로 고양이 입니다
>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
'''