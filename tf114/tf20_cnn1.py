import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical  # 1부터 시작
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10]) # 원핫인코딩 해줬으므로

#2. 모델구성
w1 = tf.get_variable('w1', shape = [2, 2, 1, 64])  # ==> (kernel_size(2,2), 채널, output(=filters, 64))  # 위의 shape와 맞아야하므로 4차원으로 써줬다. #여기서는 1이니까 흑백
L1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'VALID')   # Layer # 위의 shape와 맞아야하므로 4차원으로 써줬다. ex) strides를 2로 넣고 싶다면 (1,2,2,1) 즉 앞 뒤는 허수(1)를 넣어준다.
        # layer = x * w1

# model.add(Conv2D(filters = 64, kernel_size = (2,2), strides = (1,1), padding = 'valid',         ====> valid <-> same
#                                     input_shape = (28,28,1)))                             
                                    
# kernel_size = 가중치였다!!!!!!

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>  
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)  # w1 * x 이므로 L1에서는 커널 사이즈가 하나 줄어져있는 상태