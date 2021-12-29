# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 _temp에 넣고 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    #width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    shear_range = 0.7,
    fill_mode= 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

augment_size = 50000

# print(x_train[0].shape) # (32, 32, 3)
# print(x_train[0].reshape(32*32*3).shape) # (3072,)
#print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1) 

randidx = np.random.randint(x_train.shape[0], size = augment_size) 

# print(randidx) # [ 8416 25031 47870 ... 45663   777 25006]
# print(np.min(randidx), np.max(randidx)) # 0 49999

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  
# print(x_augmented.shape) # (50000, 32, 32, 3)
# print(y_augmented.shape) # (50000, 1)

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,   #np.zeros(augument_size), 
                                  batch_size= augment_size, shuffle= False) #save_to_dir='../_temp/').next()[0] #save_to_dir='../_temp/').next()[0]

#print(x_augmented)
# print(x_augmented.shape)  # (50000, 32, 32, 3)

# concatenate, merge 합치기 위한 것!

x_train = np.concatenate((x_train, x_augmented))  # concatenate는 괄호는 2개 써줘야한다.
y_train = np.concatenate((y_train, y_augmented))  
#print(x_train.shape, y_train.shape)  # (100000, 32, 32, 3) (100000, 1)

#print(np.unique(y_train)) 

#2. 모델
model = Sequential()
model.add(Conv2D(30,(2,2), activation='relu', input_shape= (32,32,3)))
model.add(Conv2D(20,(2,2),activation='relu'))
model.add(Conv2D(10,(2,2), activation= 'relu'))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=5, batch_size=100, validation_split=0.25, callbacks=[es]) 

#4)평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)

from sklearn.metrics import accuracy_score  
accuracy = accuracy_score(y_test, y_pred)
print('acc score:', accuracy)

''' 
loss: [3.6078758239746094, 0.16820000112056732]
acc score: 0.1682
'''








