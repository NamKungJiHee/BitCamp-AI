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
model.add(Dense(100,input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min', verbose=1, restore_best_weights=True) 

'''
 restore_best_weights = 
 True: training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함
 False: 마지막 training이 끝난 후의 weight로 놔둔당..

Epoch 227/1000
323/323 [==============================] - 1s 4ms/step - loss: 27.6073 - val_loss: 33.1006
Restoring model weights from the end of the best epoch.
Epoch 00227: early stopping
걸린시간:  353.339 초
4/4 [==============================] - 0s 4ms/step - loss: 16.5271
loss:  16.52708625793457
r2스코어 :  0.8022672322033525

restore_best_weights를 사용할 시 최적의 weight값을 기록만 할 뿐 저장 기능은 없다!
저장하기 위해서는 ModelCheckpoint라는 함수는 사용해야함!
'''


start= time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])   #여기서 loss & val_loss의 결과가 나옴  #es를 호출할거닷!
end= time.time()- start

print("걸린시간: ", round(end,3), '초')  #소수 3까지만 출력

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)



import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#print("=========================")
#print(hist)  
#print("=========================")
#print(hist.history)  
#print("=========================")
#print(hist.history['loss'])  
print("=========================")
print(hist.history['val_loss']) 



plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid() #격자표시
plt.title('loss')
plt.ylabel('loss') #y축
plt.xlabel('epoch') #x축
plt.legend(loc='upper right') 
plt.show()