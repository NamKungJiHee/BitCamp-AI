import tensorflow as tf
#print(tf.__version__)  # 1.14.0
#print('hello world')

hello = tf.constant('Hello World') # constant(상수: 고정된 수, 변하지 않는 수) // variable(변수: 변할 수 있는 수)
                                   # 대문자로 잡으면 주로 상수 / 소문자는 주로 변수
                                   # tf.placeholder = 집어넣는 것
                                   # 연산결과를 보고 싶다면 sess.run(op)을 통과시켜야 한다!!!

# print(hello) # Tensor("Const:0", shape=(), dtype=string)

#sess = tf.Session() # 구버전
sess = tf.compat.v1.Session() # session을 무조건 해줘야 한다 = 머신을 만들어준다는 것
print(sess.run(hello)) # b'Hello World'