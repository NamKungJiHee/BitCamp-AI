from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D,Reshape,Conv1D,LSTM
from tensorflow.keras.datasets import mnist

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1,        
                 padding='same', input_shape=(28, 28, 1)))   #(28,28,10)  
model.add(MaxPooling2D())   # (14,14,10)
model.add(Conv2D(5, (2,2), activation = 'relu'))     # 13,13,5  # 4차원
model.add(Conv2D(7, (2,2), activation='relu'))    # 12,12,7
model.add(Conv2D(7, (2,2), activation='relu'))    # 11,11,7
model.add(Conv2D(10, (2,2), activation='relu'))    # 10,10,10
#model.add(Flatten())   # N,1000  # 2차원
model.add(Reshape((100,10)))    # (None, 100, 10)  # 3차원
#model.add(Reshape(target_shape=(100,10)))    # (None, 100, 10)  # 상동
model.add(Conv1D(5,2))    # (None, 99, 5)
model.add(LSTM(15))       # Conv1D와 lstm모두 3차원이므로 여기서는 Flatten을 안해줘도 된다. --> 2차원으로 output
model.add(Dense(10,activation= 'softmax'))       
                   
model.summary()

""" 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 5)         205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 7)         147
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 7)         203
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 10)        290
_________________________________________________________________
reshape (Reshape)            (None, 100, 10)           0
_________________________________________________________________
conv1d (Conv1D)              (None, 99, 5)             105
_________________________________________________________________
lstm (LSTM)                  (None, 15)                1260
_________________________________________________________________
dense (Dense)                (None, 10)                160
=================================================================
Total params: 2,420
Trainable params: 2,420
"""
