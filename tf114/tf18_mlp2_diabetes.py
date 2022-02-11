from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([10,90]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([90]), name = 'bias1')

Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([90,80]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([80]), name = 'bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([80,70]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([70]), name = 'bias3')

Hidden_layer3 = tf.nn.selu(tf.matmul(Hidden_layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([70,50]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([50]), name = 'bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([50,40]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([40]), name = 'bias5')

Hidden_layer5 = tf.matmul(Hidden_layer4,w5) + b5


w6 = tf.compat.v1.Variable(tf.random.normal([40,10]), name = 'weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([10]), name = 'bias6')

Hidden_layer6 = tf.matmul(Hidden_layer5,w6) + b6

w7 = tf.compat.v1.Variable(tf.random.normal([10,1]), name = 'weight7')
b7 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias7')

#2. 모델 구성
hypothesis = tf.matmul(Hidden_layer6, w7) + b7 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w7, b7],
                                         feed_dict = {x : x_train,
                                                      y:y_train})
    if epochs % 2000== 0:
        print(epochs, loss_val,)

#4. 평가, 예측
y_pred = tf.matmul(x, w7) + b7
y_pred_data = sess.run(y_pred, feed_dict={x : x_test})
# print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test , y_pred_data)
print('r2 :', r2)

# r2 : -3.6478472768224846
# 기존 r2: 0.6108767513773117