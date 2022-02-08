# y = wx + b
############# keras01.1.py와 같은 것 #############
import tensorflow as tf
tf.set_random_seed(66)  # random_state

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(1, dtype = tf.float32) # tensorflow2에서 초기 weight는 랜덤
b = tf.Variable(1, dtype = tf.float32) # tensorflow2에서 bias는 통상적으로 0

#2. 모델구성
hypothesis = x_train * w + b # y = wx + b // model.add(Dense)와 같은 것

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse(다 더하고 n으로 나눠주기) # hypothesis = y_predict/ y_test 
                                                        # square=제곱
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법 
train = optimizer.minimize(loss)  # loss값을 최소화 시키기(optimizer로) // # model.compile(loss = 'mse', optimizer = 'sgd')와 같은 것

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)  # train이 최종적으로 모든 계산된 값을 담고 내려왔기 때문에 sess.run은 train을 해준다.
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b)) # GradientDescentOptimizer에 의해서 값이 갱신된다.
        
# sess.close()     
'''
learning_rate = 0.1 # 학습률
gradient = tf.reduce_mean((W * X - Y) * X) # d/dW
descent = W - learning_rate * gradient #경사하강법
update = W.assign(descent) # 업데이트
'''