import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(66)

#1. 데이터
datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1,1)
#print(x.shape, y.shape)  #  (178, 13) (178,)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.8, random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 3])

w1 = tf.compat.v1.Variable(tf.random.normal([13,10]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name = 'bias1')

Hidden_layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([10,30]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([30]), name = 'bias2')

Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([30,40]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([40]), name = 'bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2,w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([40,50]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([50]), name = 'bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([50,3]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([3]), name = 'bias5')

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer4, w5) + b5)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.00001).minimize(loss)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, loss_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, loss_val)
 
#4. 평가, 예측           
    y_acc_test = sess.run(tf.argmax(y_test, 1))  # 실제값
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))  # 예측값
    accuracy = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", accuracy)

# 기존 acc: 0.8333333134651184
# acc:  0.3888888888888889