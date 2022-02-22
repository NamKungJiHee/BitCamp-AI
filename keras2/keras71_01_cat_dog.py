from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')
from tensorflow.keras.applications.xception import preprocess_input

#1. 데이터
path = '../_data/image/catdog/cat_dog/'       

# save 구간 

'''
img_datagen = ImageDataGenerator(rescale= 1/255.)
catdog_train = img_datagen.flow_from_directory(      
    path + '/training_set/',
    target_size = (100, 100),                                                                       
    batch_size=100000,                                   
    class_mode='binary',  
    classes= ['cats','dogs']        
)   
catdog_test = img_datagen.flow_from_directory(         
    path + '/test_set/',
    target_size=(100,100),
    batch_size=1000000,
    class_mode='binary',  
    classes= ['cats','dogs']                          
) 
np.save(path + '/training_set/catdog_train_x', arr=catdog_train[0][0])    
np.save(path + '/training_set/catdog_train_y', arr=catdog_train[0][1])    
np.save(path + '/test_set/catdog_test_x', arr=catdog_test[0][0])      
np.save(path + '/test_set/catdog_test_y', arr=catdog_test[0][1]) 
'''
# load구간 
x_train = np.load('../_data/_save_npy/keras48_1_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_1_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_1_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_1_test_y.npy')

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # (8005, 100, 100, 3) (8005,) (2023, 100, 100, 3) (2023,)

x_train = preprocess_input(x_train)  
x_test = preprocess_input(x_test)
print("===================preprocess_input(x)=======================")
print(x_train.shape, x_test.shape)

#2. 모델링
xception = Xception(weights = 'imagenet', include_top= False, input_shape= (100, 100, 3))         

model = Sequential()
model.add(xception)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1, activation = 'sigmoid'))
    
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))
''' 
모델명 : vgg19
Time : 5.2878
loss : 0.6487
Acc : 0.6

===========================   

모델명 : xception 
Time : 5.8964
loss : 0.9577
Acc : 0.6  
=================preprocess_input=================
learning_rate:  0.0001
loss:  0.7009
accuracy:  0.4
걸린시간:  8.3212
'''