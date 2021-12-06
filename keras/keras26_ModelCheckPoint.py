from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
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
  
#model.load_weights('./_save/keras25_1_save_weights.h5')     

#model.save("./_save/keras25_1_save_model.h5")  
#model.save_weights("./_save/keras25_1_save_weights.h5")


#model.load_weights("./_save/keras25_3_save_weights.h5")  

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1) #restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',verbose=1, save_best_only=True, filepath = './_ModelCheckPoint/keras26_1_MCP.hdf5')  #다 저장하면 되므로 patience 필요 없음
                                                                                 # save_best_only: 
                                                                                #checkpoint는 Earlystopping과 쓰는게 good!
                                                                                 # patience 값을 많이 주면 그만큼 checkpoint를 많이 하게 된다!!
                                                                                 #하지만 너무 patience값을 많이 주면 그만큼 자원낭비의 위험이 있따!
                                                            #filepath = './_ModelCheckPoint'  checkpoint를 여기에 저장하랏!
                                                                #checkpoint = 최소 loss값 저장!

start= time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp]) 
end= time.time()- start


print("=========================")
print(hist)  #자료형
print("=========================")
print(hist.history)  #딕셔너리 / epoch당 loss, epochs당 val
print("=========================")
print(hist.history['loss'])  #(보기편하게 : loss값)
print("=========================")
print(hist.history['val_loss']) #(보기편하게 : val_loss값)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid() #격자표시
plt.title('loss')
plt.ylabel('loss') #y축
plt.xlabel('epoch') #x축
plt.legend(loc='upper right') 
plt.show()

print("걸린시간: ", round(end,3), '초')  

'''
{'loss': [422.4117126464844, 153.6649932861328, 134.3312530517578, 128.84767150878906, 87.1712417602539, 80.75225067138672, 75.05513000488281, 
78.35163879394531, 68.63835144042969], 'val_loss': [481.4419250488281, 76.26959991455078, 103.67216491699219, 84.1969985961914, 68.4779281616211, 
63.384124755859375, 76.44969940185547, 87.56715393066406, 78.51909637451172]}

patience가 3이므로 끝에서부터 3뺀 것 즉 63.38412475585937가 가장 좋은(낮은) loss지점!   <최소의 loss= 최적의 weight>

'''


model.save("./_save/keras26_1_save_model.h5")
#model.save_weights("./_save/keras25_3_save_weights.h5")


#model.load_weights('./_save/keras25_1_save_weights.h5')    
#model.load_weights("./_save/keras25_3_save_weights.h5")     
        

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

"""
restore_best_weights=True 했을 때 



restore_best_weights=True 뺐을 때

loss:  40.94005584716797
r2스코어 :  0.5101864231100468

"""