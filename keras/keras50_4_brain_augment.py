<<<<<<< HEAD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
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
    '../_data/image/brain/train/', 
    target_size = (150,150), 
    batch_size=160,
    class_mode = 'binary',
    shuffle= True) 
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150,150),
    batch_size = 160,
    class_mode = 'binary') 
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0] 
y_test = xy_test[0][1]
# print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160,)
# print(x_test.shape, y_test.shape) # (120, 150, 150, 3) (120,)

augment_size = 5000 
randidx = np.random.randint(x_train.shape[0], size = augment_size)

# print(x_train.shape[0])   # 160
# print(randidx) # [61 134  74 ...  60  38 117]
# print(np.min(randidx), np.max(randidx)) # 0 159

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augmented.shape) # (5000, 150, 150, 3)
# print(y_augmented.shape) # (5000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0] #save_to_dir='../_temp/').next()[0]  # 증폭

#print(x_augmented)
# print(x_augmented.shape)  # (5000, 150, 150, 3)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (5160, 150, 150, 3) (50160,)
#print(np.unique(y_train)) # [0. 1.]

#2. 모델
model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (150,150,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

#hist = model.fit_generator(xy_train, epochs= 200, steps_per_epoch= 32, validation_data= xy_test,validation_steps= 4, callbacks = [es])  
model.fit(x_train, y_train, epochs=10, batch_size=20, validation_split=0.2, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)                      

''' 
loss:  [0.7747269868850708, 0.5333333611488342]
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
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
    '../_data/image/brain/train/', 
    target_size = (150,150), 
    batch_size=160,
    class_mode = 'binary',
    shuffle= True) 
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size = (150,150),
    batch_size = 160,
    class_mode = 'binary') 
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0] 
y_test = xy_test[0][1]
# print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160,)
# print(x_test.shape, y_test.shape) # (120, 150, 150, 3) (120,)

augment_size = 5000 
randidx = np.random.randint(x_train.shape[0], size = augment_size)

# print(x_train.shape[0])   # 160
# print(randidx) # [61 134  74 ...  60  38 117]
# print(np.min(randidx), np.max(randidx)) # 0 159

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augmented.shape) # (5000, 150, 150, 3)
# print(y_augmented.shape) # (5000, )

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False).next()[0] #save_to_dir='../_temp/').next()[0]  # 증폭

#print(x_augmented)
# print(x_augmented.shape)  # (5000, 150, 150, 3)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (5160, 150, 150, 3) (50160,)
#print(np.unique(y_train)) # [0. 1.]

#2. 모델
model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (150,150,3)))
model.add(Conv2D(20, (2,2), padding='same'))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

#hist = model.fit_generator(xy_train, epochs= 200, steps_per_epoch= 32, validation_data= xy_test,validation_steps= 4, callbacks = [es])  
model.fit(x_train, y_train, epochs=10, batch_size=20, validation_split=0.2, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)                      

''' 
loss:  [0.7747269868850708, 0.5333333611488342]
>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
'''