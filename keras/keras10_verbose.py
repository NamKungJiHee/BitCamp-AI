from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt   #그림
from sklearn.model_selection import train_test_split
import time

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])


#x_train, x_test, y_train, y_test = train_test_split(x,y,
         #train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(44, input_dim=1))
model.add(Dense(11))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(44))
model.add(Dense(44))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=3)
end = time.time() - start  # 시간이 얼마나 걸렸는지..
print("걸린시간 : ", end)
# 0 : 걸린시간 :  2.8673346042633057
# 1 : 걸린시간 :  3.9569196701049805
# 2 : 걸린시간 :  3.1837007999420166
# 3 : 걸린시간 :  3.049102783203125

'''
verbose
0: 화면에 안보임
1: 화면에 보임
2: loss까지(process 과정 안보고 싶을 때)
3~: epochs만 나옴
'''

'''
#4. 평가, 예측
loss = model.evaluate(x, y)   
print('loss : ', loss)


y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict) 
print('r2스코어 : ', r2)
'''


