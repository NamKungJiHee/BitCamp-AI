from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#1) 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target  
#print(x.shape, y.shape)   # (569, 30) (569,)

#print(np.unique(y))   # [0 1]  : 이진분류

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

print(x_train.shape, x_test.shape)   # (398, 30) (171, 30)

scaler = RobustScaler() #scaler = MinMaxScaler() #scaler = StandardScaler() #scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(398,30,1)
x_test = scaler.transform(x_test).reshape(171,30,1)



#2) 모델링

model = Sequential()
model.add(LSTM(80, input_shape = (30,1))) 
model.add(Dense(45))
model.add(Dense(1, activation ='sigmoid'))


#3. 컴파일, 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs = 1000, validation_split=0.2, callbacks=[es], batch_size = 50)


#4 평가예측
loss = model.evaluate(x_test,y_test)

"""  
-LSTM-
loss: 0.0923 - accuracy: 0.9532
-CNN-
loss :  0.06962944567203522
accuracy :  0.7007591818555365
-DNN-
loss:  [0.3342837691307068, 0.8508771657943726]
"""




