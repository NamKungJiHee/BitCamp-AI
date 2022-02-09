import tensorflow as tf 
import matplotlib.pyplot as plt 
tf.compat.v1.set_random_seed(66)

x_train = [1,2,3]
y_train = [1,2,3]

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

w_history = []
loss_history = []

for step in range(21):
    # sess.run(update, feed_dict = {x:x_train, y:y_train})
    # print(step, '\t', sess.run(loss, feed_dict = {x: x_train, y: y_train}), sess.run(w))
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x: x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
    
sess.close()
    
print("================w_history================")
print(w_history)
print("================loss_history================")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel("Weight")
plt.ylabel("loss")
plt.show()