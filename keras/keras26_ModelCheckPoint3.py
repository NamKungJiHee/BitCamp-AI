from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


model = Sequential()

model.add(Dense(40,input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=False, filepath = './_ModelCheckPoint/keras26_3_MCP.hdf5') 
                                                            
                                                                
start= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp]) 
end= time.time()- start


print("걸린시간: ", round(end,3), '초')  


model.save("./_save/keras26_3_save_model.h5")

#model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')     #ES과 ModelCheckPoint를 씀으로써 여기에는 가장 좋은 weight들이 저장됨!
  



#4. 평가, 예측



print("============================1. 기본 출력=======================")

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("============================2. load_model 출력=======================")

model2 = load_model('./_save/keras26_3_save_model.h5')  

loss2= model2.evaluate(x_test, y_test)
print('loss2: ', loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)


print("============================3. ModelCheckPoint 출력=======================")

model3 = load_model('./_ModelCheckPoint/keras26_3_MCP.hdf5')  

loss3= model3.evaluate(x_test, y_test)
print('loss3: ', loss3)

y_predict3 = model3.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)

"""
loss:  108.52605438232422
r2스코어 :  -0.2984235713617629
============================2. load_model 출력=======================
4/4 [==============================] - 0s 562us/step - loss: 108.5261
loss2:  108.52605438232422
r2스코어 :  -0.2984235713617629
============================3. ModelCheckPoint 출력=======================
4/4 [==============================] - 0s 3ms/step - loss: 30.4572
loss3:  30.457210540771484
r2스코어 :  0.6356049098448162

"""




