import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.random.normal([2,1], name = 'weight'))
b = tf.compat.v1.Variable(tf.random.normal([1], name = 'bias'))      
# random_normal : 0~1 사이의 정규확률분포 값을 생성해주는 함수
# random_uniform : 0~1 사이의 균등확률분포 값을 생성해주는 함수

#2. 모델구성
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0004)
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
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype = tf.float32))     
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})    
print("=========================")
#print("예측값: \n", hy_val)
#print("예측결과: \n", pred)
print("Accuracy: ", acc)

sess.close() 
# Accuracy:  0.75