import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# 분류, 회귀모델

#1) 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))   # [0 1 2] = 다중분류      #이걸 해봐야 이진분류인지 다중분류인지 알 수 있움!!

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)   #One Hot Encoding
# print(y)
# print(y.shape)   # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)



#2) 모델구성 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(10, activation = 'linear', input_dim = 4))     
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(40))
# model.add(Dense(50))
# model.add(Dense(3, activation = 'softmax'))
model = LinearSVC() # 이거는 onehotencoding을 안해줘도 된다.

#3) 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   # accuracy는 훈련에 영향 안끼친당

# es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) 

# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

model.fit(x_train, y_train) # 훈련시켜줌

#4) 평가, 예측
# loss=model.evaluate(x_test, y_test)  
# print('loss: ', loss[0]) 
# print('accuracy: ', loss[1]) 
result = model.score(x_test, y_test)  # 결과가 accuracy로 나옴 # metrics의 개념(accuracy값을 돌려준다.)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)   #(,) 값 비교

print("result: ", result)
print("accuracy: ", acc)

""" 
result:  0.9666666666666667
accuracy:  0.9666666666666667
"""