import tensorflow as tf
tf.set_random_seed(77)  

#1. 데이터   
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape = [None])  
y_train = tf.placeholder(tf.float32, shape = [None]) 

w = tf.Variable(tf.random_normal([1], dtype = tf.float32))
b = tf.Variable(tf.random_normal([1], dtype = tf.float32))

#2. 모델구성
hypothesis = x_train * w + b # y = wx + b  

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  
                                                        
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) 
train = optimizer.minimize(loss)  

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

loss_val_list = []
W_val_list = []

for step in range(2001):
    #sess.run(train) 
    _, loss_val, W_val, b_val = sess.run([train, loss, w, b], feed_dict = {x_train:x_train_data, y_train:y_train_data}) 
    
    if step % 20 == 0:
        print(step, loss_val, W_val, b_val)
    
    loss_val_list.append(loss_val)    
    W_val_list.append(W_val)    
        
#4. 예측
x_test_data = [6,7,8]
x_test = tf.placeholder(tf.float32, shape = [None])  

predict = x_test * W_val + b_val  # y_predict = model.predict와 같다.
print("[6,7,8] 예측", sess.run(predict, feed_dict = {x_test:x_test_data})) 

sess.close() 

import matplotlib.pyplot as plt
plt.plot(loss_val_list[100:])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()