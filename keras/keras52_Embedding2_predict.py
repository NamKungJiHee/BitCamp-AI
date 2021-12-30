from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밌어요', '참 최고에요','참 잘 만든 영화에요', '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요', '글쎄요',             # x (13,?)
        '별로에요', '생각보다 지루해요', '연기가 어색해요',' 재미없어요', '너무 재미없다', '참 재밌네요', '선생님이 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])   # y (13,) [0,1]=이진분류

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, 
# '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '선생님이': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x) # [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]  
### shape 맞춰주려면 가장 긴 것을 기준으로 공백을 채워주면 된다!!! ###  공백채울 때 보통 0을 써주는데 중간에 넣어주면 안된다. 
### 앞이나 뒤에 채워주면 되는데 통상적으로 앞에 채워준다. ###

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', maxlen=5)  # 최대길이에 맞추려고 5  [11, 12, 13, 14, 15]
print(pad_x)
print(pad_x.shape)  # (13, 5) 

word_size = len(token.word_index)
print('word_size: ', word_size)  # word_size: 27  (이것은 pad_sequences 해주기 전 / pad_sequences로 해주면서 0 추가 해줬으므로 28)

print(np.unique(pad_x))  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding  # Embedding: 좌표로 change

### OneHotEncoding으로 하면 (13,5) --> (13, 5, 28)로 change ###
# 옥스포드 사전은? (13,5,1000000) --> 대략 6500만개.. : 이렇게 하면 안된다.

#2. 모델
model = Sequential()
#                                                     인풋은 (13,5)
#                     단어사전 갯수  (숫자 바꿔도 됨)   단어수, 길이(열의 갯수)
#model.add(Embedding(input_dim = 27, output_dim= 10, input_length=5)) # Embedding: 원핫인코딩안한것을 벡터화해주는것 / Embedding에서만 input_dim이 output이 아니다. 통상적으로는 그 자리가 output 갯수 / output_dim이 출력 갯수
#model.add(Embedding(27,10,input_length=5))
model.add(Embedding(27,10))  # 단어 사전의 갯수만 알아도 돌아간다!!!!(단어 사전 갯수보다 작으면 에러!!)
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

#model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs = 100, batch_size=32)   # labels = y

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]   #[0]은 loss
print('acc: ', acc)

###### 실습 #####  긍정인가 부정인가    (1)
"""
x_predict = '나는 반장이 정말 재미없다 정말 '

real_x_predict = [x_predict]
#print(real_x_predict)  # list로 만들어주기

token = Tokenizer()
token.fit_on_texts(real_x_predict)
#print(token.word_index)  # {'정말': 1, '나는': 2, '반장이': 3, '재미없다': 4}

x_predict2 = token.texts_to_sequences(real_x_predict)
#print(x_predict2)    # [[2, 3, 1, 4, 1]]

pad_x2 = pad_sequences(x_predict2, padding='pre', maxlen=5)  
print(pad_x2)   # [[ 0  0  0  0 23]]
# # print(pad_x2.shape)   # (1, 5)

final_predict = model.predict(pad_x2)

print(final_predict)
"""
#실습######################################소스를 완성하거라!   (2)
#결과는??? 긍정인지? 부정인지?
x_predict = '나는 반장이 정말 재미없다 정말'
x_predict = [x_predict]                         # 리스트로 만들기
#print(x_predict)

token = Tokenizer()
token.fit_on_texts(x_predict)   
# print(token.word_index)
# {'정말': 1, '나는': 2, '반장이': 3, '재미없다': 4}
#print(x_predict)

x_predict = token.texts_to_sequences(x_predict)
#print(len(x_predict))

pad_x_predict = pad_sequences(x_predict, padding='pre')
y_pred = model.predict(pad_x_predict)
print(y_pred)

if y_pred < 0.5:
    print('부정')
else : 
    print('긍정')


""" 
pad_x_predict = pad_sequences(x_predict, padding='pre')
y_pred = model.predict(pad_x_predict)
print(y_pred)
score = float(model.predict(pad_x_predict)) # 예측

if y_pred < 0.5:
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
else : 
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
"""
