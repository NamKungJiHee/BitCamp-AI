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
                                                        
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.175) 
train = optimizer.minimize(loss)  

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    #sess.run(train) 
    _, loss_val, W_val, b_val = sess.run([train, loss, w, b], feed_dict = {x_train:x_train_data, y_train:y_train_data}) 
    
    if step % 20 == 0:
        #print(step, sess.run(loss), sess.run(w), sess.run(b)) 
        print(step, loss_val, W_val, b_val)
    
########################### 실습 ###########################
#4. 예측
x_test_data = [6,7,8]
x_test = tf.placeholder(tf.float32, shape = [None])  

predict = x_test * W_val + b_val  # y_predict = model.predict와 같다.
print("[6,7,8] 예측", sess.run(predict, feed_dict = {x_test:x_test_data}))  # 13,15,17이 나와야 한다!!

sess.close() 
# lr수정해서 epoch를 100번 이하로 줄여라!
# step = 100 이하, w = 1.9999, b = 0.9999
''' 
0 90.204796 [5.621887] [2.192736]
20 8.012792 [3.0978687] [1.3133907]
40 0.7119116 [2.3350108] [1.0755891]
60 0.0632765 [2.1031528] [1.0149888]
80 0.005628967 [2.0321448] [1.0012763]
100 0.00050160766 [2.010174] [0.99902976]
[6,7,8] 예측 [13.060075 15.070249 17.080421]
'''