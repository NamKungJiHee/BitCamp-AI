import autokeras as ak
#from tensorflow.keras.datasets import mnist
import tensorflow as tf

#1. 데이터
(x_train, y_train), (x_test, y_test) = \
                              tf.keras.datasets.cifar100.load_data()

#2. 모델
model = ak.ImageClassifier(overwrite = True,
                           max_trials = 5)   # 모델을 5번 돌리겠다. 즉 epochs = 10로 주었으니 총 50번 도는 것! 이 중 좋은 것을 쓰겠다.

#3. 컴파일, 훈련
model.fit(x_train, y_train, epochs = 10)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print("results: ", results)
# results:   [2.606215715408325, 0.34220001101493835] # loss와 accuracy값  # max_trials = 2, epoches = 5로 주었을 때
# results:    # loss와 accuracy값  # max_trials = 5, epoches = 10로 주었을 때