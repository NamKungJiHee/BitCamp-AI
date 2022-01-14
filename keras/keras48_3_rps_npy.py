
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
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
    fill_mode= 'nearest')   
 
test_datagen = ImageDataGenerator(rescale=1./255) 

xy_train = train_datagen.flow_from_directory(
    '../_data/image/rps/rps/', 
    target_size = (100,100), 
    batch_size=10,
    class_mode = 'categorical',
    shuffle= True) 
#Found 2520 images belonging to 1 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/rps/rps/',
    target_size = (100,100),
    batch_size = 10,
    class_mode = 'categorical') 
#Found 2520 images belonging to 1 classes.

# np.save('../_data/_save_npy/keras48_3_train_x.npy', arr = xy_train[0][0])  
# np.save('../_data/_save_npy/keras48_3_train_y.npy', arr = xy_train[0][1])
# np.save('../_data/_save_npy/keras48_3_test_x.npy', arr = xy_test[0][0])
# np.save('../_data/_save_npy/keras48_3_test_y.npy', arr = xy_test[0][1])

x_train = np.load('../_data/_save_npy/keras48_3_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_3_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_3_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_3_test_y.npy')

# print(x_train.shape)  # (10, 100, 100, 3)
# print(y_train.shape)  # (10,1)
# print(x_test.shape)  # (10, 100, 100, 3)
# print(y_test.shape)  # (10,1)

#2. 모델링
model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (100,100,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

model.fit(x_train,y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
