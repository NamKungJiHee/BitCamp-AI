<<<<<<< HEAD
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.python.keras.backend import dropout

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

# print(x_train.shape, x_test.shape)  # (25000,) (25000,)
# print(y_train.shape, y_test.shape)  # (25000,) (25000,)
# print(np.unique(y_train))  # [0 1]

#print(x_train[0], y_train[0])

# print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# print(len(x_train[0]), len(x_train[1])) # 218 189
# print(type(x_train[0]), type(x_train[1])) # <class 'list'> <class 'list'>   
# print(max(len(i) for i in x_train)) # 2494
# print(sum(map(len, x_train))/len(x_train))  # 238.71

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding= 'pre', maxlen= 200, truncating= 'pre') 
                                                                                 
#print(x_train.shape)  # (25000, 200)

x_test = pad_sequences(x_test, padding= 'pre', maxlen= 200, truncating= 'pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape, y_train.shape) # (25000, 200) (25000, 2)
#print(x_test.shape, y_test.shape) # (25000, 200) (25000, 2)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout

model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim= 20, input_length=200)) 
model.add(LSTM(32))  
model.add(Dense(20))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size=10)   

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc: ', acc)

''' 
782/782 [==============================] - 4s 5ms/step - loss: 3.1165 - acc: 0.8436
acc:  0.8436400294303894
=======
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.python.keras.backend import dropout

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

# print(x_train.shape, x_test.shape)  # (25000,) (25000,)
# print(y_train.shape, y_test.shape)  # (25000,) (25000,)
# print(np.unique(y_train))  # [0 1]

#print(x_train[0], y_train[0])

# print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# print(len(x_train[0]), len(x_train[1])) # 218 189
# print(type(x_train[0]), type(x_train[1])) # <class 'list'> <class 'list'>   
# print(max(len(i) for i in x_train)) # 2494
# print(sum(map(len, x_train))/len(x_train))  # 238.71

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding= 'pre', maxlen= 200, truncating= 'pre') 
                                                                                 
#print(x_train.shape)  # (25000, 200)

x_test = pad_sequences(x_test, padding= 'pre', maxlen= 200, truncating= 'pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape, y_train.shape) # (25000, 200) (25000, 2)
#print(x_test.shape, y_test.shape) # (25000, 200) (25000, 2)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Dropout

model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim= 20, input_length=200)) 
model.add(LSTM(32))  
model.add(Dense(20))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size=10)   

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc: ', acc)

''' 
782/782 [==============================] - 4s 5ms/step - loss: 3.1165 - acc: 0.8436
acc:  0.8436400294303894
>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
'''