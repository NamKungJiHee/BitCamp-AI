from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

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

#D:\_data\image\catdog\cat_dog\

xy_train = train_datagen.flow_from_directory(
    '../_data/image/horseorhuman/horse-or-human/', 
    target_size = (100,100), 
    batch_size=2000,
    class_mode = 'binary',
    shuffle= True
) 
#target_size = (150,150) 괄호 안의 숫자는 내 마음대로 정할 수 있다. = 맞추고 싶은 사이즈로 
#"categorical"은 2D형태의 원-핫 인코딩된 라벨, "binary"는 1D 형태의 이진 라벨입니다, "sparse"는 1D 형태의 정수 라벨, "input"은 인풋 이미지와 동일한 이미지
#######Found 160 images belonging to 2 classes.#######

xy_test = test_datagen.flow_from_directory(
    '../_data/image/horseorhuman/horse-or-human/',
    target_size = (100,100),
    batch_size = 2000,
    class_mode = 'binary'
    ) 
#####Found 120 images belonging to 2 classes.#####

# print(xy_train)   #x와 y가 함께 있음
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E743074F40>

print(xy_train[0][0].shape, xy_train[0][1].shape)  # x = (1027, 100, 100, 3) (1027,)
#print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>

# np.save('./_save_npy/keras48_2_train_x.npy', arr = xy_train[0][0])   
# np.save('./_save_npy/keras48_2_train_y.npy', arr = xy_train[0][1])
# np.save('./_save_npy/keras48_2_test_x.npy', arr = xy_test[0][0])
# np.save('./_save_npy/keras48_2_test_y.npy', arr = xy_test[0][1])

x_train = np.load('./_save_npy/keras48_2_train_x.npy')
y_train = np.load('./_save_npy/keras48_2_train_y.npy')
x_test = np.load('./_save_npy/keras48_2_test_x.npy')
y_test = np.load('./_save_npy/keras48_2_test_y.npy')

# print(x_train.shape)  # (1027, 100, 100, 3)
# print(y_train.shape)  # (1027,)
# print(x_test.shape)  # (1027, 100, 100, 3)
# print(y_test.shape)  # (1027,)

#2. 모델링
model = Sequential()
model.add(Conv2D(30, (2,2), input_shape= (100,100,3)))
model.add(Conv2D(20, (2,2)))  
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)

model.fit(x_train,y_train, epochs = 100, validation_split=0.2, callbacks=[es], batch_size = 50)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

''' 
loss:  [0.0, 1.0]

'''





