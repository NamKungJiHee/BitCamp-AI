import numpy as np 
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]   # XOR Data

#2. 모델
#model = LinearSVC()  # 선 그어서 분류
#model = Perceptron()
#model = SVC() # polynomial (다항식)  
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

#3. 훈련
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs = 100)

#4. 평가, 예측

y_predict = model.predict(x_data)

results = model.evaluate(x_data, y_data)  # x_data와 y_data의 accuracy

print(x_data,"의 예측 결과: ", y_predict)
print("metrics_acc: ", results[1])

# acc = accuracy_score(y_data, np.round(y_predict,0)) # 실제값, 예측값
# print("accuracy_score: ", acc)

"""
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과:  [[0.51677936]
 [0.3010964 ]
 [0.55599403]
 [0.33530107]]
metrics_acc:  0.5                 --> 반올림
"""
