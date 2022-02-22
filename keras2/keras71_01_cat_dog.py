from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')

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

model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid')]

for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    model.add(Flatten())  
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.0001)  
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=15, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=50,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=1)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=50,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")
    
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
'''