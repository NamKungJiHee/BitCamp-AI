from unittest import result
import numpy as np 
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[1, 2, 1, 1],     # (8, 4)
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],  # 2    # (8, 3)
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],  # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],  # 0
          [1, 0, 0]]

x_predict = [[1, 11, 7, 9]]  # (1,4) -> (N, 4)


x = tf.compat.v1.placeholder(tf.float32, shape = [None, 4]) 
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3]) 
 
w = tf.compat.v1.Variable(tf.random.normal([4,3]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name = 'bias')   # y가 (8, 3) 즉 3이기 때문에 b에 (1,3) 넣어준다.

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)  # neural network
# model.add(Dense(3, activation='softmax'))

#3-1. 컴파일
#loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
#loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(loss)  

sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sees:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
            
    results = sess.run(hypothesis, feed_dict = {x: x_predict})    
    print(results, sess.run(tf.arg_max(results, 1)))  

# [[9.3190324e-01 6.8059169e-02 3.7637248e-05]] [0]  <== 0번째가 가장 크다는 뜻