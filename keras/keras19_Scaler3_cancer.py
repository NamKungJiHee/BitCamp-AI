from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target  # y: 0 or 1
#print(x.shape, y.shape)   # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)  # 어느 비율로 나눌지
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train에 맞는 비율로 들어가있움

model = Sequential()
model.add(Dense(10, input_dim = 30)) 
model.add(Dense(20))   
model.add(Dense(30,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(50))
model.add(Dense(1, activation = 'sigmoid'))

#model.summary()

#3) 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) # auto, max, min

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2) #callbacks=[es])

#4) 평가, 예측

#회귀모델의 보조지표로 r2를 썼다면 이진분류에서는 필요가 없다..
loss=model.evaluate(x_test, y_test)  
print('loss: ', loss) 






"""
# 결과       loss/accuracy
# 그냥
loss:  [0.3342837691307068, 0.8508771657943726]

# MinMAX
loss:  [0.10253071039915085, 0.9824561476707458]

#Standard
loss:  [0.17529627680778503, 0.9649122953414917]

# robust 
loss:  [0.15628808736801147, 0.9561403393745422]

# maxabs
loss:  [0.11401195079088211, 0.9824561476707458]

== MinMax Scaler로 돌렸을 때 가장 최소의 loss 값이 나온다! 네가지 방법의 scaler 모두 loss의 값이 낮게 나오고 그냥 돌렸을 때는 확실히 loss의 값이
크게 나옴을 알 수 있따!

##################################### relu 적용 후 
# 그냥
loss:  [0.1779232621192932, 0.9473684430122375]

# MinMAX
loss:  [0.21782004833221436, 0.9298245906829834]

#Standard
loss:  [0.29931432008743286, 0.9649122953414917]
----------------------------------------------------'relu함수를 위쪽에 넣고 돌렸는데 전보다 결과가 좋지 않다!
# robust 
loss:  [0.8295993208885193, 0.9385964870452881]

# maxabs
loss:  [0.31733039021492004, 0.9473684430122375] 
----------relu함수를 중간에 넣고 돌림
== 앞선 diabets도 그렇고 cancer도 그렇고 relu를 중간에 넣고 나면 loss값이 너무 안좋아짐!!


######################summary

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                310
_________________________________________________________________
dense_1 (Dense)              (None, 20)                220
_________________________________________________________________
dense_2 (Dense)              (None, 30)                630
_________________________________________________________________
dense_3 (Dense)              (None, 40)                1240
_________________________________________________________________
dense_4 (Dense)              (None, 50)                2050
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 51
=================================================================
Total params: 4,501
Trainable params: 4,501
Non-trainable params: 0














"""
