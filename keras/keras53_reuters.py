from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train,y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)  # num_words : 단어 사전의 갯수

print(x_train, len(x_train), len(x_test)) #  8982   2246
print(y_train[0])  # 3
print(np.unique(y_train))  # 46개의 뉴스 카테고리
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape)  # (8982,) (8982,)

print(len(x_train[0]), len(x_train[1])) # 87,56
print(type(x_train[0]), type(x_train[1])) # <class 'list'> <class 'list'>       ################# 즉 numpy 안에 list가 들어가있다. ###################
#print("뉴스기사의 최대길이: ", max(len(x_train))) #  error
print("뉴스기사의 최대길이: ", max(len(i) for i in x_train)) # 뉴스기사의 최대길이:  2376
print("뉴스기사의 평균길이: ", sum(map(len, x_train))/len(x_train))  # map : x_train의 len(조건)을 출력해주는 것 즉 8982의 전체 길이를 반환해준다. (전체/합)  # 뉴스기사의 평균길이:  145.53

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding= 'pre', maxlen= 100, truncating= 'pre') # 최대길이는 2376이지만 너무 낭비이므로 그냥 평균에서 조금 줄인 100으로 하겠다. 
                                                                                 #truncating: if maxlen보다 길이가 크면 앞에서부터 절단하겠다. 
print(x_train.shape)  # (8982, 2376) --> (8982,100) b/c 최대길이가 2376이므로

x_test = pad_sequences(x_test, padding= 'pre', maxlen= 100, truncating= 'pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape) # (2246, 100) (2246, 46)

##############################################################
word_to_index = reuters.get_word_index()
#print(word_to_index)
#print(sorted(word_to_index.items())) # key위주로 나옴
import operator
print(sorted(word_to_index.items(), key = operator.itemgetter(1)))  # dictionary: key(0), value(1)
# == value, key 순으로 나온다.

# Ex) ('the', 1), ('of', 2), ('to', 3), ('in', 4), ('said', 5)

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token

print(' '.join([index_to_word[index] for index in x_train[0]]))

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim= 10, input_length=100)) 
model.add(LSTM(32))  # kernel size = 2,2
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size=10)   

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]  
print('acc: ', acc)

''' 
71/71 [==============================] - 1s 5ms/step - loss: 5.8731 - acc: 0.6162
acc:  0.6162065863609314
'''

