from sklearn.datasets import load_boston, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time


#datasets = load_boston()
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min', verbose=1, restore_best_weights= True)


'''
restore_best_weights 사용
True: training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함
False: 마지막 training이 끝난 후의 weight로 놔둔당..

Epoch 51/1000
282/282 [==============================] - 1s 4ms/step - loss: 3097.3494 - val_loss: 3083.3835
Restoring model weights from the end of the best epoch.
Epoch 00051: early stopping
걸린시간:  62.691 초
3/3 [==============================] - 0s 5ms/step - loss: 3244.6689
loss:  3244.6689453125
r2스코어 :  0.5000541433580152

restore_best_weights를 사용할 시 최적의 weight값을 기록만 할 뿐 저장 기능은 없다!
저장하기 위해서는 ModelCheckpoint라는 함수는 사용해야함!

'''



start= time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es]) 
end= time.time()- start

print("걸린시간: ", round(end,3), '초')

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


'''
hist= [3747.350830078125, 3381.92431640625, 3259.57861328125, 4003.77685546875, 3110.82861328125, 3508.234130859375, 3233.7880859375, 3290.865234375, 3207.021240234375, 3201.530517578125, 3120.232421875, 3200.121826171875, 3185.6064453125, 3220.714111328125, 3305.96923828125, 3018.152587890625, 3349.95751953125, 3658.191650390625, 3693.1171875, 3145.81103515625, 3001.50537109375, 4546.64990234375, 3039.408203125, 3565.33935546875, 3993.728759765625, 3248.469970703125, 3520.717529296875, 3110.397216796875, 3043.276611328125, 3217.71240234375, 3124.313232421875, 3087.41943359375, 3354.500244140625, 3140.19970703125, 3451.085693359375, 3052.197509765625, 3299.798095703125, 3197.5869140625, 3270.8662109375, 3605.572265625, 3240.660888671875, 3100.842529296875, 3142.458984375, 3314.907470703125, 3406.9189453125, 3136.791748046875, 3103.572021484375, 3083.937744140625, 3544.086669921875, 3089.178466796875, 3083.383544921875]

restore_best_weights 사용
True: training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원함
False: 마지막 training이 끝난 후의 weight로 놔둔당..
Epoch 51/1000
282/282 [==============================] - 1s 4ms/step - loss: 3097.3494 - val_loss: 3083.3835
Restoring model weights from the end of the best epoch.
Epoch 00051: early stopping
걸린시간:  62.691 초
3/3 [==============================] - 0s 5ms/step - loss: 3244.6689
loss:  3244.6689453125
r2스코어 :  0.5000541433580152
restore_best_weights를 사용할 시 최적의 weight값을 기록만 할 뿐 저장 기능은 없다!

'''

