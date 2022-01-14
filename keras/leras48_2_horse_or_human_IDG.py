<<<<<<< HEAD
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

# 1. 데이터

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

train_generator = train_datagen.flow_from_directory('../_data/image/horseorhuman/horse-or-human/',target_size=(50,50),
    batch_size=10,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horseorhuman/horse-or-human/', # same directory as training data
    target_size=(50,50),
    batch_size=10,
    class_mode='categorical',
    subset='validation') # set as validation data   == test와 같은것.

print(train_generator[0][0].shape)  
print(validation_generator[0][0].shape) 

test_datagen = ImageDataGenerator(rescale = 1./255)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(128,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
hist = model.fit_generator(train_generator, epochs = 1000, steps_per_epoch = 72,  
                    validation_data = validation_generator,
                    validation_steps = 4, callbacks=[es])

model.save("./_save/keras48_2_save_weights1111.h5")

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

pic_path = '../_data/image/horseorhuman/predict/my_picture.jpg'
model_path = './_save/keras48_2_save_weights1111.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(50,50))
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
    print(pred)
    horse = pred[0][0]*50
    human = pred[0][1]*50
    if horse > human:
        print(f"당신은 {round(horse,2)} % 확률로 말 입니다")
    else:
        print(f"당신은 {round(human,2)} % 확률로 사람 입니다")

''' 
loss: 884212992.0
val_loss: 53916468.0
acc: 0.48678719997406006
val_acc: 1.0
[[0. 1.]]
당신은 50.0 % 확률로 사람 입니다
'''



=======
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

# 1. 데이터

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

train_generator = train_datagen.flow_from_directory('../_data/image/horseorhuman/horse-or-human/',target_size=(50,50),
    batch_size=10,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horseorhuman/horse-or-human/', # same directory as training data
    target_size=(50,50),
    batch_size=10,
    class_mode='categorical',
    subset='validation') # set as validation data   == test와 같은것.

print(train_generator[0][0].shape)  
print(validation_generator[0][0].shape) 

test_datagen = ImageDataGenerator(rescale = 1./255)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(128,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
hist = model.fit_generator(train_generator, epochs = 1000, steps_per_epoch = 72,  
                    validation_data = validation_generator,
                    validation_steps = 4, callbacks=[es])

model.save("./_save/keras48_2_save_weights1111.h5")

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

pic_path = '../_data/image/horseorhuman/predict/my_picture.jpg'
model_path = './_save/keras48_2_save_weights1111.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(50,50))
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
    print(pred)
    horse = pred[0][0]*50
    human = pred[0][1]*50
    if horse > human:
        print(f"당신은 {round(horse,2)} % 확률로 말 입니다")
    else:
        print(f"당신은 {round(human,2)} % 확률로 사람 입니다")

''' 
loss: 884212992.0
val_loss: 53916468.0
acc: 0.48678719997406006
val_acc: 1.0
[[0. 1.]]
당신은 50.0 % 확률로 사람 입니다
'''



>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
