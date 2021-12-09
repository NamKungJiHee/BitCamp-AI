from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
import numpy as np 
from sklearn.model_selection import train_test_split

#1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape)  # (506,13)  -> (506,13,1,1)
# print(y.shape)  # (506, )

print(datasets.feature_names)  #컬럼 이름
# print(datasets.DESCR)

xx = pd.DataFrame(x, columns= datasets.feature_names)   #   CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT                     
#print(type(xx))                                        # x가 numpy상태인데 그것을 pandas로 바꿔준다
#print(xx)                                              # <class 'pandas.core.frame.DataFrame'>  / pandas는 여러가지 자료형을 넣을 수 있다.

#print(xx.corr())    # 상관관계

xx['price'] = y

#print(xx)
print(xx.corr())

import matplotlib.pyplot as plt
import seaborn as sns    # matplot보다 이뿌게

# plt.figure(figsize=(10,10))
# sns.heatmap(data = xx.corr(), square = True, annot=True, cbar=True)   #cbar = 옆에 컬러
# plt.show()


x = xx.drop(['CHAS','price'],axis=1)  # (506,12)
x = x.to_numpy()
#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) # 앞에서 조금 걸러내고 이제 분리시켜서 훈련,테스트

scaler = MinMaxScaler()


n = x_train.shape[0]  # ==> 아까 (506,13)이었는데 그 중에 앞에 506
x_train_transe = scaler.fit_transform(x_train)    #scaler 작업 (scaler는 2차원밖에 안되므로 먼저 scaler부터 해주고 차원을 바꿔준다)
x_train = x_train_transe.reshape(n,2,2,3)    # 2차원을 3,4차원으로 바꿔줘야함 / 아까 하나 버렸으니까 (506,12)이므로 12로 만들어준다! (n,12,1,1)도 가능!

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,2,3)

#print(x_train.shape)


#2) 모델링

model = Sequential()
model.add(Conv2D(100, kernel_size=(2,2),padding ='same', strides=1, input_shape = (2,2,3)))
model.add(MaxPooling2D())
model.add(Conv2D(45,(2,2),padding ='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(45))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam') 

es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)


"""   
loss :  7.7401814460754395
R2 :  0.906312596952948

"""