# import tensorflow as tf
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Conv2D

# tf.compat.v1.set_random_seed(66)

# #1. 데이터
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# from keras.utils import to_categorical  # 1부터 시작
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
# x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
# y = tf.compat.v1.placeholder(tf.float32, [None, 10]) 

# #2. 모델구성

# # Layer1
# w1 = tf.get_variable('w1', shape = [2, 2, 1, 16])  
# L1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'SAME')   
#         # layer = x * w1
# L1 = tf.nn.relu(L1)
# L1_maxpool = tf.nn.max_pool2d(L1, ksize= [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')                       

# print(w1) 
# print(L1) 
# print(L1_maxpool)

# # Layer2   
# w2 = tf.get_variable('w2', shape = [3, 3, 16, 64])                                             
# L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1, 1, 1, 1], padding = 'SAME')
# L2 = tf.nn.selu(L2)
# L2_maxpool = tf.nn.max_pool2d(L2, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
# print(L2_maxpool)

# # Layer3
# w3 = tf.get_variable('w3', shape = [3, 3, 64, 32])
# L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1, 1, 1, 1], padding = 'SAME')
# L3 = tf.nn.elu(L3)
# L3_maxpool = tf.nn.max_pool2d(L3, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
# print(L3_maxpool) 

# Layer4
# w4 = tf.get_variable('w4', shape = [3, 3, 32, 16], initializer= tf.contrib.layers.xavier_initializer()) # weight를 초기화시키는것
# L4 = tf.nn.conv2d(L3_maxpool, w4, strides = [1, 1, 1, 1], padding = 'SAME')
# L4 = tf.nn.elu(L4)
# L4_maxpool = tf.nn.max_pool2d(L4, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')
# print(L4_maxpool) 

# # Flatten
# L_flat = tf.reshape(L3_maxpool, [-1, 2*2*32])
# print("Flatten: ", L_flat) 

# # Layer5 DNN

# w5 = tf.get_variable('w5', shape = [2*2*32,64], initializer = tf.contrib.layers.xavier_initializer())

# b5 = tf.Variable(tf.random_normal([64]), name = 'b5')
# L5 = tf.matmul(L_flat, w5) + b5
# L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob = 0.7)
# print(L5) 

# # Layer6 DNN
# w6 = tf.get_variable('w6', shape = [64,32], initializer = tf.contrib.layers.xavier_initializer())

# b6 = tf.Variable(tf.random_normal([32]), name = 'b6')
# L6 = tf.matmul(L5, w6) + b6
# L6 = tf.nn.relu(L6)
# L6 = tf.nn.dropout(L6, keep_prob = 0.7)
# print(L6)

# # Layer7 Softmax
# w7 = tf.get_variable('w7', shape = [32,10], initializer = tf.contrib.layers.xavier_initializer())

# b7 = tf.Variable(tf.random_normal([10]), name = 'b7')
# L7 = tf.matmul(L6, w7) + b7
# hypothesis = tf.nn.softmax(L7)
# print(hypothesis) 

# # 3-1. 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) # categorical_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())

# training_epochs = 10
# batch_size = 100
# total_batch = int(len(x_train)/batch_size)
# print(total_batch)

# # 3-2 훈련
# for epoch in range(training_epochs):
#     avg_loss = 0
    
#     for i in range(total_batch):   # 600번 돈다.
#         start = i * batch_size     # 0 
#         end = start + batch_size   # 100
#         batch_x, batch_y = x_train[start:end], y_train[start:end] # 0~100
        
#         feed_dict = {x:batch_x, y:batch_y}   # 단순변수
        
#         batch_loss, _ = sess.run([loss, optimizer],feed_dict = feed_dict)
        
#         avg_loss += batch_loss / total_batch
        
#     print('Epoch: ', '%04d' %(epoch + 1), 'loss: , {:9f}'.format(avg_loss))
    
# print("훈련 끝") 

# prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
# print('ACC:', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))

#########################################################################################################################

from pickletools import optimize
from unittest import result
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

tf.compat.v1.set_random_seed(66)

# 1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델
#layer1
w1 = tf.get_variable("w1", shape=[2, 2, 1, 32]) # 
#<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#<tf.Tensor 'Conv2D_1:0' shape=(?, 28, 28, 64) dtype=float32>
#print(L1)
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L1_maxpool) # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

#layer2
w2 = tf.get_variable("w2", shape=[3, 3, 32, 16])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME') 
# Tensor("Conv2D_2:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L2_maxpool) # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

#layer3
w3 = tf.get_variable("w3", shape=[3, 3, 16, 8])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#layer4
w4 = tf.get_variable("w4", shape=[3, 3, 8, 16])
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(L4_maxpool)

#Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*16])
print(L_flat) # Tensor("Reshape:0", shape=(?, 128), dtype=float32)

#layer5 DNN
w5 = tf.get_variable("w5", shape=[2*2*16, 64], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name='b1')
l5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
l5 = tf.nn.dropout(l5, keep_prob=0.7)
print(l5) # Tensor("dropout/mul:0", shape=(?, 64), dtype=float32)

#layer6 DNN
w6 = tf.get_variable("w6", shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name='b1')
l6 = tf.nn.relu(tf.matmul(l5, w6) + b6)
l6 = tf.nn.dropout(l6, keep_prob=0.7)
print(l6) # Tensor("dropout_1/mul:0", shape=(?, 32), dtype=float32)

#layer7 softmax
w7 = tf.get_variable("w7", shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]), name='b1')
hypothesis = tf.nn.softmax(tf.matmul(l6, w7) + b7)
print(hypothesis) # Tensor("add_1:0", shape=(?, 10), dtype=float32)

#3. 손실함수, 최적화, 예측
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#4. 훈련
for epochs in range(20):
    avg_loss = 0
    total_batch = int(x_train.shape[0] / 100) # 60000 / 100 = 600
    for i in range(total_batch): # 600번 반복
        batch_xs, batch_ys = x_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100] # 100개씩 잘라서 배치
        feed_dict = {x: batch_xs, y: batch_ys} 
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict) # 훈련
        avg_loss += batch_loss / total_batch # 평균 손실값

        correct_prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(y, 1))        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Epoch:', '%04d' % (epochs + 1), 'loss =', '{:.9f}'.format(avg_loss), 'accuracy', accuracy.eval(session=sess, feed_dict={x: x_test, y: y_test}))

#5. 평가

print = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
print('Accuracy:', accuracy)


# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)), tf.float32)) # 정확도
# predict = tf.math.argmax(hypothesis, 1) # 예측값
# print('Accuracy:', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) # 손실함수
# print('Loss:', sess.run(loss, feed_dict={x: x_test, y: y_test}))
'''
Accuracy: 0.9758
'''