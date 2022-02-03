import autokeras as ak
#from tensorflow.keras.datasets import mnist
import tensorflow as tf

# autokeras.com 참고

#1. 데이터
(x_train, y_train), (x_test, y_test) = \
                              tf.keras.datasets.mnist.load_data()

#2. 모델
model = ak.ImageClassifier(overwrite = True,
                           max_trials = 2)   # 모델을 2번 돌리겠다. 즉 epochs = 5로 주었으니 총 10번 도는 것! 이 중 좋은 것을 쓰겠다.

#3. 컴파일, 훈련
model.fit(x_train, y_train, epochs = 5)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print("results: ", results)
# results:  [0.02564852684736252, 0.9922000169754028] # loss와 accuracy값

#model.summary()