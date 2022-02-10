# sigmoid 사용하기 #
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569,1)
#print(x_data.shape, y_data.shape) # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape = [None,30])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.random.normal([30,1]), name = 'weight')     
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
#hypothesis =  tf.matmul(x,w) + b
hypothesis = tf.sigmoid(tf.matmul(x,w) + b) 

#3-1. 컴파일
#loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, 'loss:', loss_val, '\n', hy_val)
    
#4. 예측
y_predict = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
# print(y_predict) # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict ={x:x_data, y:y_data}))  # [False] , [True]
# print(sess.run(y_predict, feed_dict ={x:x_data, y:y_data}))  # [0.] , [1.]
# print(sess.run(tf.equal(y, y_predict), feed_dict ={x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype = tf.float32))  

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})    
print("=========================")
print("예측값: \n", hy_val)
print("예측결과: \n", pred)
print("Accuracy: ", acc)

sess.close() 
''' 
Accuracy:  0.37258348
'''