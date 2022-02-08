import tensorflow as tf
print(tf.__version__) # 1.14.0  # 2.7.0
print(tf.executing_eagerly()) # False   # True: tensorflow2의 환경에서 실행시킨 것
# 즉시 실행 모드: tensorflow2에서 tensorflow1의 sess.run을 쓸 수 있다.

tf.compat.v1.disable_eager_execution() # 즉시 실행 모드 off
print(tf.executing_eagerly()) # False

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))