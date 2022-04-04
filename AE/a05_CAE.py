# CNN으로 딥하게 구성

# Conv2D
# MaxPool
# Conv2D
# MaxPool
# Conv2D    --> encoder

# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D    --> decoder

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float')/255
x_test = x_test.reshape(10000, 28,28,1).astype('float')/255

#2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D (hidden_layer_size, kernel_size = (2,2), strides = 1, input_shape = (28,28,1), activation = 'relu'))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(784, (2,2), activation = 'relu'))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(784, (2,2), activation = 'relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(784, (2,2), activation = 'relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(784, (2,2), activation = 'relu'))
    return model

# UpSampling = 데이터가 실제보다 더 자주 수집된 것처럼 데이터를 표현하는 것
# 업샘플링 목적: 드물게 측정된 데이터에서 더 조밀한 시간의 데이터를 얻기 위함

model = autoencoder(hidden_layer_size= 32)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, x_train, epochs = 10)

#4. 평가, 예측
output = model.predict(x_test)

# import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2,5,figsize = (20,7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28,1), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28,1), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()      