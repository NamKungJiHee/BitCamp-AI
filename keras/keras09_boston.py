from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)  #컬럼 이름
print(datasets.DESCR)

model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=50, batch_size=5)

loss = model.evaluate(x, y)   
print('loss : ', loss)


y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)  
print('r2스코어 : ', r2)

#0.8이상
#loss :  23.494508743286133
#r2스코어 :  0.7216934825017536
