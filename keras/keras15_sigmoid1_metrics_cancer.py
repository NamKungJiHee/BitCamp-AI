import numpy as np 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#1) 데이터
datasets = load_breast_cancer() #암 걸렸낭 안걸렸낭
#print(datasets) #data 보여줌
#print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target  # y: 0 or 1
print(x.shape, y.shape)   # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



#print(y_test[:11])
#print(y[:10])
#print(y)    #(0과 1만 뜬당: 이진분류닷!)
#print(np.unique(y))   # [0 1]   =  y는 0과 1로만 이루어져있따..


#2) 모델구성
model = Sequential()
model.add(Dense(10, activation = 'linear', input_dim = 30))  #activation은 layer에 들어간당.. 'linear'는 명시하지 않아도 들어가있는것..
model.add(Dense(20, activation = 'sigmoid'))   #만약 앞에 수치가 크다고 생각한다면 sigmoid를 써줘도 된당!
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(1, activation = 'sigmoid'))

#3) 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose=1, restore_best_weights=True) # auto, max, min

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4) 평가, 예측

#회귀모델의 보조지표로 r2를 썼다면 이진분류에서는 필요가 없다..
loss=model.evaluate(x_test, y_test)  
print('loss: ', loss)  #loss가 2개 출력된다

results=model.predict(x_test[:31])
print(y_test[:31])    #[1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1]
print(results)

'''
###시그모이드###
#1) 실행 
Epoch 22/100
364/364 [==============================] - 0s 562us/step - loss: 0.6642 - accuracy: 0.6236 (loss의 정확도) - val_loss: 0.6610 - val_accuracy: 0.6264 (val_loss의 정확도)
Restoring model weights from the end of the best epoch.
Epoch 00022: early stopping
4/4 [==============================] - 0s 665us/step - loss: 0.6536 - accuracy: 0.6404

#2) 실행
4/4 [==============================] - 0s 0s/step - loss: 0.3581 - accuracy: 0.8947
loss:  [0.3580891788005829, 0.8947368264198303]
[1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1 1 1]
[[0.9639791 ]
 [0.9639796 ]
 [0.963979  ]
 [0.9639796 ]
 [0.9638753 ]
 [0.96388674]
 [0.02844474] ...
 
#################loss:  [0.6536130905151367, 0.640350878238678]       # 앞에꺼는 전체 최종 loss /진짜 loss(binary_crossentropy), 뒤에꺼는 metrics[accuracy] ###########
default 기본값.
metrics는 훈련 상황만 출력해서 보여주는 것.. metrics는 훈련에 영향을 미치지 않는닷..
리스트 = 2개 이상 쓰일때 (2개이상 리스트)
######### callback=[es]  : []이렇게 쓰인것을 보면 다른 리스트가 있다는 것... 
좋은 모델이라는 것을 판단하는 기준: loss가 낮을수록 good
r2는 회귀모델의 보조지표.. so 이진분류에서는 쓸 이유가 x
#이진 분류를 할 때 양쪽의 데이터 값이 비슷하게 있어야 함!
회귀모델에서는 명시하지는 않았지만 activation = 'linear'가 있었음!!! 
loss=평가지표=cost=손실

q)이진분류에서 r2의 역할이 accuracy?
마지막 accuracy가 metrics[accuracy]인데 그럼 이게 전체 loss의 accuracy인가?
'''

'''
Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수

One Hot Encoding = 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여,
다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식..



'''
