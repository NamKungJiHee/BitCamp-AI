import tensorflow as tf
tf.compat.v1.set_random_seed(66)

## 히든 레이어를 2개 이상으로 늘려라!!!

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

# bias = 행렬의 덧셈
# 입력 값은 placeholder
# zeros는 통상적으로 bias에 넣는다.

# Input Layer
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2,30]), name = 'weight1') 
b1 = tf.compat.v1.Variable(tf.zeros([30]), name = 'bias1')  

#2. 모델 구성
# Hidden_layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)  # sigmoid는 해도되고 안해도 된다. 
# Hidden_layer1 = tf.matmul(x,w1) + b1
Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([30,5]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([5]), name = 'bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([5,1]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias3')

# Hidden_layer3 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)
# w4 = tf.compat.v1.Variable(tf.random.normal([5,1]), name = 'weight4')
# b4 = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias4')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer2, w3) + b3)   # x는 상위 레이어의 아웃풋

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(101):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)
        
#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
print(sess.run(hypothesis > 0.5, feed_dict = {x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float32))
pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_data, y : y_data})

print('예측 결과 :', '\n', pred)
print('accuracy :', acc)
# accuracy : 1.0