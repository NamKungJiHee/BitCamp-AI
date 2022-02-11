import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
(x_train, y_train), (x_test,  y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (60000, 28, 28) (10000, 28, 28) (60000,) (10000,)

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (60000, 784) (10000, 784) (60000, 1) (10000, 1)

ohe = OneHotEncoder()
ohe.fit(y_train)
y_train= ohe.transform(y_train).toarray()        # 원핫인코딩 성공 #
y_test = ohe.transform(y_test).toarray()
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (60000, 784) (10000, 784) (60000, 10) (10000, 10)

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

w1 = tf.compat.v1.Variable(tf.zeros([784,128]), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([1,128]), name = 'bias')

Hidden_layer1 = tf.matmul(x,w1) + b1

w2 = tf.compat.v1.Variable(tf.random_normal([128,80]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([1,80]), name = 'bias2')

Hidden_layer2 = tf.matmul(Hidden_layer1,w2) + b2

w3 = tf.compat.v1.Variable(tf.random_normal([80,60]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([1,60]), name = 'bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2,w3) + b3

w4 = tf.compat.v1.Variable(tf.random_normal([60,40]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random_normal([1,40]), name = 'bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.random_normal([40,20]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random_normal([1,20]), name = 'bias5')

Hidden_layer5 = tf.matmul(Hidden_layer4,w5) + b5

w6 = tf.compat.v1.Variable(tf.random_normal([20,10]), name='weight6')
b6 = tf.compat.v1.Variable(tf.random_normal([1,10]), name = 'bias6')

Hidden_layer6 = tf.matmul(Hidden_layer5,w6) + b6

w7 = tf.compat.v1.Variable(tf.random_normal([10,10]), name='weight7')
b7 = tf.compat.v1.Variable(tf.random_normal([1,10]), name = 'bias7')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer6, w7) + b7) 

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        _, loss_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
 
#4. 평가, 예측           
    y_acc_test = sess.run(tf.argmax(y_test, 1))  # 실제값
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))  # 예측값
    accuracy = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", accuracy)
    
# 기존: 0.972000002861023
# acc: 0.098