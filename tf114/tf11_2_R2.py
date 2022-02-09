import tensorflow as tf 
import matplotlib.pyplot as plt 
tf.compat.v1.set_random_seed(66)

x_train_data = [1,2,3]
y_train_data = [1,2,3]
x_test_data = [4,5,6]
y_test_data = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x)  # 평균
descent = w - lr * gradient
update = w.assign(descent)  # w = w - lr * gradient  # assign 할당함수  ===> gradientdescentoptimizer

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x: x_train_data, y:y_train_data})
    print(step, '\t', loss_v, '\t', w_v)
    
    # w_history.append(w_v)
    # loss_history.append(loss_v)

############################### R2 만들기 ###############################
from sklearn.metrics import r2_score, mean_absolute_error  # pip install scikit-learn
x_test = tf.compat.v1.placeholder(tf.float32)
y_test = tf.compat.v1.placeholder(tf.float32)

y_predict = x_test * w_v
y_predict_data = sess.run(y_predict, feed_dict = {x_test: x_test_data})
print("y_predict_data: ", y_predict_data) # y_predict_data:  [3.9999902 4.9999876 5.999985 ]

r2 = r2_score(y_test_data, y_predict_data)  # 실제값, 예측값
print("r2: ", r2) # r2:  0.9999999997661178

mae = mean_absolute_error(y_test_data, y_predict_data)
print("mae: ", mae) # mae:  1.2318293253580729e-05

sess.close()