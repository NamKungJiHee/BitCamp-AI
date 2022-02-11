import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
tf.set_random_seed(66)

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x_data = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
#print(x.columns)
test_file = test_file.drop(['datetime'], axis=1)
#print(x.shape)    # (10886, 8)
y_data = train['count']
#print(y.shape)  #(10886,)
y_data = y_data.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.8, random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([8,10]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name = 'bias1')

Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([10,20]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([20]), name = 'bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([20,8]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([8]), name = 'bias3')

Hidden_layer3 = tf.nn.selu(tf.matmul(Hidden_layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([8,1]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias4')

#2. 모델 구성
hypothesis = tf.matmul(Hidden_layer3, w4) + b4 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w4, b4],
                                         feed_dict = {x : x_train,
                                                      y:y_train})
    if epochs % 2000== 0:
        print(epochs, loss_val,)

#4. 평가, 예측
y_pred = tf.matmul(x, w4) + b4
y_pred_data = sess.run(y_pred, feed_dict={x : x_test})
# print('y_pred_data :',y_pred_data)


r2 = r2_score(y_test , y_pred_data)
print('r2 :', r2)

# 기존 r2: 0.26061793730829397
# mlp r2:-1.309474563952159