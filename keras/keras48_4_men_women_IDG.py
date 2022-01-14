import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3)                   # set validation split

train_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/rps/',
    target_size=(100,100),
    batch_size=3,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/rps/', # same directory as training data
    target_size=(100,100),
    batch_size=3,
    class_mode='categorical',
    subset='validation') # set as validation data

print(train_generator[0][0].shape)  # (10, 100, 100, 3)
print(validation_generator[0][0].shape) # (10, 100, 100, 3)

test_datagen = ImageDataGenerator(rescale = 1./255)

# np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])

# print(train_generator[0])
# print(validation_generator[0])

## np.save('./_save_npy/keras48_21_train_x.npy', arr = train_generator[0][0])
## np.save('./_save_npy/keras48_21_train_y.npy', arr = train_generator[0][1])
## np.save('./_save_npy/keras48_21_test_x.npy', arr = validation_generator[0][0])
## np.save('./_save_npy/keras48_21_test_y.npy', arr = validation_generator[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(128,(2,2), input_shape = (100,100,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
hist = model.fit_generator(train_generator, epochs = 1000, steps_per_epoch = 1103, 
                    validation_data = validation_generator,
                    validation_steps = 4, callbacks=[es])

model.save("./_save/keras48_4_save_weights1111.h5")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/menwomen/jihee/my_picture.jpg'
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
    men = pred[0][0]*100
    women = pred[0][1]*100
    if men > women:
        print(f"당신은 {round(men,2)} % 확률로 남자 입니다")
    else:
        print(f"당신은 {round(women,2)} % 확률로 여자 입니다")

''' 
loss: 153426912.0
val_loss: 3787776.0
acc: 0.5226757526397705
val_acc: 0.0
당신은 96.63 % 확률로 여자 입니다
'''