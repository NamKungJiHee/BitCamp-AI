import tensorflow as tf

sess = tf.compat.v1.Session()
x = tf.Variable([2], dtype = tf.float32) # x=2라는 뜻

init = tf.compat.v1.global_variables_initializer() # 변수를 사용할 수 있게 초기화 
sess.run(init)  # '이제 이 변수를 쓸 수 있다'라는 뜻

print('x:', sess.run(x)) # x: [2.]