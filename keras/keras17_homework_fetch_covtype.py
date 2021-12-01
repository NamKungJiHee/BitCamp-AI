import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (581012, 54) (581012,)  ----> (다중분류) 
print(np.unique(y))    # [1 2 3 4 5 6 7]     --->다중분류임을 알 수 있움!

#(x)  앞에서부터 빈자리 채워줌

''' 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)   #One Hot Encoding
print(y)
print(y.shape) # (581012, 8) 원핫인코딩     #categorical은 앞에 0부터 시작 그래서 8로 나옴
'''

#1)  
# sklearn

'''
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)  #if sparse = True면 metrics로 출력, False면 array로 출력
y = ohe.fit_transform(y.reshape(-1,1))  #1부터 시작 ~ -1즉 배열 끝까지 출력!...
print(y.shape)  # (581012, 7)
'''

'''
744/744 [==============================] - 1s 753us/step - loss: 0.6517 - accuracy: 0.7165 - val_loss: 0.6519 - val_accuracy: 0.7160
3632/3632 [==============================] - 1s 359us/step - loss: 0.6514 - accuracy: 0.7181
loss:  [0.6513694524765015, 0.7181311845779419]
[[1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0.]]
results:  [[7.4306548e-01 2.2528991e-01 2.2806358e-08 3.1031550e-14 8.8607392e-04
  4.0713687e-08 3.0758465e-02]
 [9.2298411e-02 9.0696383e-01 8.9399955e-06 8.8406669e-09 5.8231334e-04
  1.1845327e-04 2.8061453e-05]
 [8.3184117e-01 1.4967537e-01 2.9472250e-07 4.1602043e-12 7.4146420e-04
  1.2938963e-05 1.7728820e-02]
 [9.4453901e-02 9.0245134e-01 8.2553684e-05 2.1020994e-06 2.3772740e-03
  2.7603158e-04 3.5680761e-04]
 [4.9688557e-01 4.8487684e-01 8.5130569e-06 5.4271704e-12 6.4864773e-03
  1.8416105e-05 1.1724131e-02]
 [3.4603006e-01 6.3115275e-01 1.0022848e-05 5.6583053e-13 2.2575341e-02
  1.2520610e-04 1.0659345e-04]
 [2.6570448e-01 7.3110247e-01 6.0402695e-06 7.2282696e-10 2.9874471e-04
  1.2412281e-05 2.8758862e-03]
 [4.6381536e-03 2.3227799e-01 5.1868254e-01 2.1042021e-04 1.5338428e-01
  9.0790302e-02 1.6299758e-05]
 [2.2864965e-01 7.5838578e-01 5.9273498e-05 3.0565082e-09 1.1849043e-02
  1.0257015e-04 9.5379632e-04]
 [1.5183393e-07 1.3022647e-04 6.8227386e-01 6.6586700e-03 5.4351534e-05
  3.1088269e-01 3.9549926e-12]
 [8.6302471e-01 1.3523042e-01 2.0950557e-08 4.6611739e-14 4.1764401e-05
  6.3570843e-07 1.7024507e-03]]
'''


#2)
# pandas
import pandas as pd   # get_dummies) 결측값이 사라져서 수가 줄어듦 (581012, 7) 일케 출력됨! / dummy_na = True) 결측값도 인코딩 처리되서 (581012, 8)일케 출력됨!
y = pd.get_dummies(y)   # pandas는 수치로 보여줌!  그 외 pandas와 sklearn은 비슷!!
print(y.shape)    #(drop_first=True)  # 열을 n-1개 생성 / (581012, 7)

'''
744/744 [==============================] - 1s 761us/step - loss: 0.6530 - accuracy: 0.7168 - val_loss: 0.6707 - val_accuracy: 0.7028
3632/3632 [==============================] - 1s 366us/step - loss: 0.6756 - accuracy: 0.7004
loss:  [0.6755886077880859, 0.7004380226135254]
        1  2  3  4  5  6  7
257457  1  0  0  0  0  0  0
15362   0  1  0  0  0  0  0
455621  1  0  0  0  0  0  0
26237   0  1  0  0  0  0  0
530518  0  1  0  0  0  0  0
2113    0  1  0  0  0  0  0
81459   0  1  0  0  0  0  0
348766  0  0  1  0  0  0  0
552259  1  0  0  0  0  0  0
239314  0  0  0  0  0  1  0
201753  1  0  0  0  0  0  0
results:  [[5.2866077e-01 4.6454406e-01 9.7635095e-08 2.4203764e-12 1.0460173e-03
  3.1928766e-07 5.7486696e-03]
 [2.5899818e-02 9.7368026e-01 1.7271488e-05 9.6159553e-08 2.0979067e-04
  1.9118718e-04 1.6121160e-06]
 [7.2244823e-01 2.7152464e-01 4.4743916e-07 6.4183803e-10 7.4133003e-04
  1.6138301e-05 5.2691768e-03]
 [3.5340350e-02 9.6297377e-01 2.8731363e-05 2.1374479e-05 1.3780734e-03
  2.4261807e-04 1.5096433e-05]
 [3.2561997e-01 6.6499299e-01 6.0457420e-05 4.0640796e-10 5.7033640e-03
  1.2166003e-04 3.5015708e-03]
 [1.9284944e-01 7.8770638e-01 5.9506751e-06 6.3981176e-10 1.9240029e-02
  1.6051011e-04 3.7737474e-05]
 [8.6482443e-02 9.1325772e-01 3.4745674e-06 9.5705470e-09 1.5900981e-04
  2.5334843e-05 7.2026276e-05]
 [1.7241886e-03 1.7090479e-01 6.6718322e-01 9.5094141e-04 3.2926336e-02
  1.2630673e-01 3.8405128e-06]
 [1.1745878e-01 8.7558877e-01 6.3211832e-04 7.2767421e-08 5.4096850e-03
  6.9420272e-04 2.1631364e-04]
 [6.1479533e-08 4.7467325e-05 7.1188641e-01 2.1627720e-02 1.4572445e-05
  2.6642379e-01 3.7960329e-12]
 [6.5959322e-01 3.4022185e-01 6.1723351e-09 1.1318654e-11 3.9599847e-05
  6.0462128e-07 1.4475231e-04]]


'''



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



#2) 모델링
model = Sequential()
model.add(Dense(10, activation= 'linear', input_dim = 54))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(7, activation = 'softmax'))



#3) 캄파일, 훈련

model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs = 100, batch_size=500, validation_split= 0.2, callbacks=[es])   # 데이터가 크기 때문에 batch_size를 크게 해줌!

#4) 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
results= model.predict(x_test[:11])
print(y_test[:11])
print('results: ', results)


'''
Q) batch_size의 디폴트는 몇??  31.9978... = 32
batch_size를 1로 했을 때 1epoch 당 371847
batch_size를 완전히 지우고 돌렸을 때 1epoch당 11621
371847/11621 = 31.9978... = 32
'''