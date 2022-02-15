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

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10]) 

#2. 모델구성

# Layer1
w1 = tf.get_variable('w1', shape = [2, 2, 1, 16])  # ==> (kernel_size(2,2), 채널, output(=filters, 16))  # 위의 shape와 맞아야하므로 4차원으로 써줬다. #여기서는 1이니까 흑백
L1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'SAME')   # Layer # 위의 shape와 맞아야하므로 4차원으로 써줬다. ex) strides를 2로 넣고 싶다면 (1,2,2,1) 즉 앞 뒤는 허수(1)를 넣어준다.
        # layer = x * w1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize= [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') # ksize = kernel_size // 맨 앞, 맨 뒤는 허수(1) 그저 shape맞춰주기 위한 것
# model.add(Conv2D(filters = 64, kernel_size = (2,2), strides = (1,1), padding = 'same',         ====> valid <-> same
#                                     input_shape = (28,28,1)))                             
                                    
# kernel_size = 가중치였다!!!!!!
print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>  
print(L1) # Tensor("Conv2D:0", shape=(?, 28, 28, 64), dtype=float32) 
print(L1_maxpool) # Tensor("MaxPool:0", shape=(?, 14, 14, 128), dtype=float32) / max_pool로 인해 반으로 줄었다.

# Layer2   
w2 = tf.get_variable('w2', shape = [3, 3, 16, 8])                                                  ### 3번째 자리는 직전 output값!!!!!! ###       
L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
print(L2_maxpool) # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# Layer3
w3 = tf.get_variable('w3', shape = [3, 3, 8, 8])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
print(L3_maxpool) # shape=(?, 4, 4, 8)

# # Layer4
# w4 = tf.get_variable('w4', shape = [3, 3, 32, 16], initializer= tf.contrib.layers.xavier_initializer()) # weight를 초기화시키는것
# L4 = tf.nn.conv2d(L3_maxpool, w4, strides = [1, 1, 1, 1], padding = 'SAME')
# L4 = tf.nn.elu(L4)
# L4_maxpool = tf.nn.max_pool2d(L4, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
# print(L4_maxpool) # shape=(?, 2, 2, 32)

# Flatten
L_flat = tf.reshape(L3_maxpool, [-1, 4*4*8])
print("Flatten: ", L_flat) # Flatten:  Tensor("Reshape:0", shape=(?, 128), dtype=float32)

# Layer5 DNN

w5 = tf.compat.v1.Variable(tf.random.uniform([128,10]), name='weight5')
Hidden_layer5 = tf.matmul(L_flat, w5) 

w6 = tf.compat.v1.Variable(tf.random.normal([10,10]), name = 'weight6')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer5, w6))
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)
                       
#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.0001).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001):
        _, loss_val = sess.run([optimizer,loss], feed_dict={x:x_train, y:y_train})
        if step % 10 ==0:
            print(step, loss_val)
    
    results = sess.run(hypothesis, feed_dict={x:x_test})
    print(results, sess.run(tf.math.argmax(results, 1))) 
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, results), dtype=tf.float32))
    pred, acc = sess.run([tf.math.argmax(results, 1), accuracy], feed_dict={x:x_test, y:y_test})

    print("예측결과 : ", pred)
    print("accuracy : ", acc)

    sess.close()
    ''' 
    예측결과 :  [5 6 8 ... 8 8 2]
    '''